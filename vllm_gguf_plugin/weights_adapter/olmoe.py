# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import gguf
import torch
from vllm.model_executor.models.utils import WeightsMapper

from ..gguf_files import GGUFModelFiles
from ..gguf_utils import maybe_patch_hf_config_from_gguf
from ..weight_utils import get_gguf_tensor_names
from .base import BaseGGUFWeightsAdapter, GGUFWeight

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.config import ModelConfig


def build_olmoe_mapper() -> WeightsMapper:
    return WeightsMapper(
        orig_to_new_prefix={
            "token_embd.": "model.embed_tokens.",
            "blk.": "model.layers.",
            "output_norm.": "model.norm.",
            "output.": "lm_head.",
        },
        orig_to_new_substr={
            "attn_q.": "self_attn.q_proj.",
            "attn_k.": "self_attn.k_proj.",
            "attn_v.": "self_attn.v_proj.",
            "attn_output.": "self_attn.o_proj.",
            "attn_q_norm.": "self_attn.q_norm.",
            "attn_k_norm.": "self_attn.k_norm.",
            "attn_norm.": "input_layernorm.",
            "ffn_norm.": "post_attention_layernorm.",
            "ffn_gate_inp.": "mlp.gate.",
            "ffn_down_exps.": "mlp.experts.0.down_proj.",
            "ffn_gate_exps.": "mlp.experts.0.gate_proj.",
            "ffn_up_exps.": "mlp.experts.0.up_proj.",
        },
    )


def split_olmoe_expert_weights(
    weights: Iterable[GGUFWeight],
) -> Iterable[GGUFWeight]:
    for name, weight in weights:
        if weight.ndim == 3 and ".experts.0." in name:
            for expert_id, expert_weight in enumerate(weight.unbind()):
                yield (
                    name.replace(
                        ".experts.0.",
                        f".experts.{expert_id}.",
                    ),
                    expert_weight,
                )
        else:
            yield name, weight


_Q8_0 = 8   # ggml type id


def _requantize_q8_0(weight: torch.Tensor, weight_type: int) -> torch.Tensor:
    """Re-encode a stacked quantized expert tensor ``[E, rows, nbytes]`` as Q8_0.

    Q8_0 keeps every 3-, 4-, 5- and 6-bit format within rounding error of its decoded
    values, so the result is as good as the original for all practical purposes."""
    from .. import ops

    E, rows, nbytes = weight.shape
    block_size, type_size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(weight_type)]
    cols = nbytes // type_size * block_size
    device = torch.device("cuda") if torch.cuda.is_available() else weight.device
    out = []
    for e in range(E):
        w = weight[e].to(device).contiguous()
        x = ops.ggml_dequantize(w, weight_type, rows, cols, torch.float32).reshape(rows, cols // 32, 32)
        d = (x.abs().amax(dim=-1, keepdim=True) / 127.0).to(torch.float16)
        q = torch.round(x / d.float().clamp_min(1e-30)).clamp_(-127, 127).to(torch.int8)
        blk = torch.cat([d.view(torch.uint8).reshape(rows, cols // 32, 2), q.view(torch.uint8)], dim=-1)
        out.append(blk.reshape(rows, cols // 32 * 34).cpu())
    return torch.stack(out)


def unify_expert_gate_up_types(
    weights: Iterable[GGUFWeight],
) -> Iterable[GGUFWeight]:
    """Give the gate and up experts of a layer the same quantization type.

    vLLM fuses both into one parameter with a single row size, but mixed-precision
    GGUFs (e.g. ByteShape) may quantize them differently in some layers. In that
    case both are re-encoded as Q8_0 (only those layers, a few hundred MB in total)."""
    held: dict[str, list] = {}   # layer prefix -> [gate_type, gate_weight]
    gate_type: dict[str, GGUFWeight] = {}
    for name, weight in weights:
        if ".experts.0.gate_proj.weight_type" in name:
            gate_type[name.split(".experts.0.")[0]] = (name, weight)
            continue
        if ".experts.0.gate_proj.weight" in name and weight.ndim == 3:
            key = name.split(".experts.0.")[0]
            held[key] = [gate_type.pop(key), (name, weight)]
            continue
        if ".experts.0.up_proj.weight_type" in name:
            key = name.split(".experts.0.")[0]
            if key in held:
                held[key].append((name, weight))
                continue
        if ".experts.0.up_proj.weight" in name and weight.ndim == 3:
            key = name.split(".experts.0.")[0]
            if key in held:
                (gtn, gt), (gn, gw), (utn, ut) = held.pop(key)
                g_t, u_t = int(gt), int(ut)
                if g_t != u_t:
                    gw, weight = _requantize_q8_0(gw, g_t), _requantize_q8_0(weight, u_t)
                    gt = ut = torch.tensor(_Q8_0)
                yield gtn, gt
                yield gn, gw
                yield utn, ut
                yield name, weight
                continue
        yield name, weight
    for key, pend in held.items():   # a layer without an up tensor: pass through unchanged
        yield from pend


class OLMoEGGUFAdapter(BaseGGUFWeightsAdapter):
    """Adapter for OLMoE-style GGUF MoE models (OLMoE, Mellum, Qwen3-MoE).

    These store each layer's experts as stacked ``ffn_{gate,up,down}_exps`` tensors,
    while vLLM loads them per expert, so the stacks are split along the expert axis.
    """

    @classmethod
    def matches(cls, config) -> bool:
        return config.model_type in ["olmoe", "mellum", "qwen3_moe"]

    def patch_hf_config(
        self,
        files: GGUFModelFiles,
        hf_config: PretrainedConfig,
    ):
        return maybe_patch_hf_config_from_gguf(files.primary_backbone, hf_config)

    def build_name_map(
        self,
        files: GGUFModelFiles,
        model_config: ModelConfig,
    ) -> dict[str, str]:
        del model_config
        mapper = build_olmoe_mapper()
        gguf_names = sorted(get_gguf_tensor_names(files.backbone))
        hf_names = mapper.apply_list(gguf_names)
        return dict(zip(gguf_names, hf_names, strict=True))

    def transform_weights(
        self,
        weights: Iterable[GGUFWeight],
        model_config: ModelConfig,
    ) -> Iterable[GGUFWeight]:
        del model_config
        yield from split_olmoe_expert_weights(unify_expert_gate_up_types(weights))
