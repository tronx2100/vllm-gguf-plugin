# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from functools import partial

import gguf

import torch
from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
)
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import direct_register_custom_op

from .. import ops
from .params import (
    GGUFUninitializedWeightParameter,
    GGUFUninitializedWeightTypeParameter,
    _gguf_moe_weight_loader,
    _gguf_moe_weight_type_loader,
)
from .utils import DEQUANT_TYPES, MMQ_QUANT_TYPES, MMVQ_QUANT_TYPES, logger

# Optional (off by default): from this many tokens on, the experts of a layer are dequantized to dense weights and run through
# vLLM's tensor-core grouped GEMM (Triton fused_experts); a big batch touches nearly every expert anyway and the int8 dp4a kernel
# behind ggml_moe_a8 is several times slower. It makes the prefill of MoE models up to ~1.8x faster, but the temporary dense
# weights raise the peak memory of a step (~0.2 GiB per GPU with the default group size, more with bigger groups), which the
# memory profile takes away from the KV cache - so it is only enabled by setting GGUF_MOE_DEQUANT_MIN_TOKENS (e.g. 1024).
MOE_DEQUANT_MIN_TOKENS = int(os.environ.get("GGUF_MOE_DEQUANT_MIN_TOKENS", 10**12))
MOE_DEQUANT_CHUNK_BYTES = int(float(os.environ.get("GGUF_MOE_DEQUANT_CHUNK_MB", "64")) * 2**20)


def _dequantize_experts(w: torch.Tensor, weight_type: int, dtype: torch.dtype) -> torch.Tensor:
    """[E, rows, bytes] quantized expert weights -> dense [E, rows, cols]."""
    E, rows, nbytes = w.shape
    block_size, type_size = gguf.GGML_QUANT_SIZES[weight_type]
    cols = nbytes // type_size * block_size
    return ops.ggml_dequantize(w.reshape(E * rows, nbytes), weight_type, E * rows, cols, dtype).view(E, rows, cols)


def _fused_moe_dequantized(x, w1, w2, topk_weights, topk_ids, weight_type, weight_type2, activation_enum):
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    # The dequantized experts of a whole layer need ~1 GB per GPU on top of the model, which the memory profile
    # counts against the KV cache. Dequantize them in groups of at most MOE_DEQUANT_CHUNK_BYTES and run the grouped
    # GEMM per group (expert_map hides the experts of the other groups; their rows are zeroed by vLLM).
    E, rows1, nbytes1 = w1.shape
    _, rows2, nbytes2 = w2.shape
    bs1, ts1 = gguf.GGML_QUANT_SIZES[weight_type]
    bs2, ts2 = gguf.GGML_QUANT_SIZES[weight_type2]
    per_expert = (rows1 * (nbytes1 // ts1 * bs1) + rows2 * (nbytes2 // ts2 * bs2)) * x.element_size()
    group = max(1, min(E, MOE_DEQUANT_CHUNK_BYTES // per_expert))
    if group >= E:
        w1d = _dequantize_experts(w1, weight_type, x.dtype)
        w2d = _dequantize_experts(w2, weight_type2, x.dtype)
        return fused_experts(x, w1d, w2d, topk_weights, topk_ids, activation=activation_enum)
    out = None
    for a in range(0, E, group):
        b = min(E, a + group)
        w1d = _dequantize_experts(w1[a:b], weight_type, x.dtype)
        w2d = _dequantize_experts(w2[a:b], weight_type2, x.dtype)
        expert_map = torch.full((E,), -1, dtype=torch.int32, device=x.device)
        expert_map[a:b] = torch.arange(b - a, dtype=torch.int32, device=x.device)
        y = fused_experts(
            x, w1d, w2d, topk_weights, topk_ids, activation=activation_enum,
            global_num_experts=E, expert_map=expert_map,
        )
        out = y if out is None else out.add_(y)
    return out


def _fused_moe_gguf(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    weight_type: int,
    weight_type2: int,
    activation: str,
) -> torch.Tensor:
    activation_enum = MoEActivation.from_str(activation)

    def act(inp: torch.Tensor):
        d = inp.shape[-1] // 2
        output_shape = inp.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=inp.dtype, device=inp.device)
        apply_moe_activation(activation_enum, out, inp)
        return out

    from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

    if (
        x.shape[0] >= MOE_DEQUANT_MIN_TOKENS
        and weight_type in DEQUANT_TYPES
        and weight_type2 in DEQUANT_TYPES
    ):
        return _fused_moe_dequantized(
            x, w1, w2, topk_weights, topk_ids, weight_type, weight_type2, activation_enum
        )

    out_hidden_states = torch.empty_like(x)
    if (
        weight_type2 in MMQ_QUANT_TYPES
        and weight_type in MMQ_QUANT_TYPES
        and x.shape[0] > 64
    ):
        num_tokens, _ = x.shape
        E, N, _ = w1.shape
        top_k = topk_ids.shape[1]
        block_size = ops.ggml_moe_get_block_size(weight_type)

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, block_size, E
        )
        out = ops.ggml_moe_a8(
            x,
            w1,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            weight_type,
            N,
            top_k,
            num_tokens,
        )
        out = act(out)
        out = ops.ggml_moe_a8(
            out,
            w2,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            weight_type2,
            w2.shape[1],
            1,
            num_tokens * top_k,
        )
        out = out.reshape(num_tokens, top_k, w2.shape[1]).mul_(
            topk_weights.view(num_tokens, top_k, 1)
        )
        ops.moe_sum(out, out_hidden_states)
    elif weight_type2 in MMVQ_QUANT_TYPES and weight_type in MMVQ_QUANT_TYPES:
        num_tokens, _ = x.shape
        E, N, _ = w1.shape
        top_k = topk_ids.shape[1]

        out = ops.ggml_moe_a8_vec(x, w1, topk_ids, top_k, weight_type, N, num_tokens)
        out = act(out)

        out = ops.ggml_moe_a8_vec(
            out, w2, topk_ids, 1, weight_type2, w2.shape[1], num_tokens * top_k
        )
        out = out.reshape(num_tokens, top_k, w2.shape[1]).mul_(
            topk_weights.view(num_tokens, top_k, 1)
        )
        ops.moe_sum(out, out_hidden_states)
    else:
        from . import fused_mul_mat_gguf as fused_mul_mat_gguf_op

        logger.warning_once(
            "There is no support for fast MoE kernel "
            "for current quantization method. "
            "Falling back to slow implementation. "
        )
        for tok, (w, idx) in enumerate(zip(topk_weights, topk_ids)):
            inp = x[tok].reshape((1,) + x.shape[1:])
            current_hidden_state = None
            for ww, ii in zip(w, idx):
                out = fused_mul_mat_gguf_op(inp, w1[ii], weight_type)
                out = act(out)
                current_state = fused_mul_mat_gguf_op(out, w2[ii], weight_type2).mul_(
                    ww
                )
                if current_hidden_state is None:
                    current_hidden_state = current_state
                else:
                    current_hidden_state.add_(current_state)
            out_hidden_states[tok] = current_hidden_state
    return out_hidden_states


def _fused_moe_gguf_fake(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    weight_type: int,
    weight_type2: int,
    activation: str,
) -> torch.Tensor:
    del w1, w2, topk_weights, topk_ids, weight_type, weight_type2, activation
    return torch.empty_like(x)


try:
    direct_register_custom_op(
        op_name="_fused_moe_gguf",
        op_func=_fused_moe_gguf,
        fake_impl=_fused_moe_gguf_fake,
    )
    fused_moe_gguf = torch.ops.vllm._fused_moe_gguf
except AttributeError as error:
    raise error


class GGUFMoEMethod(FusedMoEMethodBase):
    """MoE method for GGUF."""

    def __init__(
        self,
        quant_config,
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del params_dtype
        base_weight_loader = extra_weight_attrs.pop("weight_loader")
        tensor_shape = (num_experts, 2 * intermediate_size_per_partition, hidden_size)
        w13_weight = GGUFUninitializedWeightParameter(requires_grad=False)
        set_weight_attrs(
            w13_weight,
            {
                "weight_loader": partial(
                    _gguf_moe_weight_loader, layer, base_weight_loader
                ),
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "data_container": [],
            },
        )
        set_weight_attrs(w13_weight, extra_weight_attrs)
        layer.register_parameter("w13_weight", w13_weight)

        w13_weight_type = GGUFUninitializedWeightTypeParameter(requires_grad=False)
        set_weight_attrs(
            w13_weight_type,
            {
                "weight_loader": _gguf_moe_weight_type_loader,
                "weight_type": 0,
                "shard_weight_type": {},
                "num_elements": 1,
                "ignore_warning": True,
            },
        )
        set_weight_attrs(w13_weight_type, extra_weight_attrs)
        layer.register_parameter("w13_weight_type", w13_weight_type)

        tensor_shape = (num_experts, intermediate_size_per_partition, hidden_size)
        w2_weight = GGUFUninitializedWeightParameter(requires_grad=False)
        set_weight_attrs(
            w2_weight,
            {
                "weight_loader": partial(
                    _gguf_moe_weight_loader, layer, base_weight_loader
                ),
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "data_container": [],
            },
        )
        set_weight_attrs(w2_weight, extra_weight_attrs)
        layer.register_parameter("w2_weight", w2_weight)

        w2_weight_type = GGUFUninitializedWeightTypeParameter(requires_grad=False)
        set_weight_attrs(
            w2_weight_type,
            {
                "weight_loader": _gguf_moe_weight_type_loader,
                "weight_type": 0,
                "shard_weight_type": {},
                "num_elements": 1,
                "ignore_warning": True,
            },
        )
        set_weight_attrs(w2_weight_type, extra_weight_attrs)
        layer.register_parameter("w2_weight_type", w2_weight_type)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        del layer
        return None

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del shared_experts, shared_experts_input
        if layer.apply_router_weight_on_input:
            raise NotImplementedError(
                "Apply router weight on input is not supported for"
                "fused GGUF MoE method."
            )

        from . import fused_moe_gguf as fused_moe_gguf_op

        expert_map = layer.expert_map
        if expert_map is not None:
            # Expert parallelism: this rank only holds a slice of the experts. Map the global ids to local ones and give
            # the experts of other ranks weight 0 (their output is added up by the all-reduce after the layer).
            local_ids = expert_map[topk_ids.long()]
            local = local_ids >= 0
            topk_weights = topk_weights * local
            topk_ids = torch.where(local, local_ids, torch.zeros_like(local_ids)).to(topk_ids.dtype)

        return fused_moe_gguf_op(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            layer.w13_weight_type.weight_type,
            layer.w2_weight_type.weight_type,
            layer.activation.value,
        )
