# SPDX-License-Identifier: Apache-2.0
"""Tiled weight layouts for the tensor-core small-batch kernels (see csrc/gguf/mmvq_tiled.cuh).

The repack is a lossless permutation done once at load time: rows are grouped in tiles of 16, the
6-bit scales are expanded to bytes and the block scales (d, dmin) are stored in a separate plane.
The tiled tensors replace the original GGUF tensor.
"""

import torch
from gguf import GGMLQuantizationType as WeightType

TILED_MAX_TOKENS = 64     # up to this many tokens the tiled mma kernel is used (measured: faster than dequantize + GEMM below ~100), above it dequantize + GEMM

# type -> bytes of the GGUF block
_BLOCK_BYTES = {
    int(WeightType.Q3_K): 110,
    int(WeightType.Q4_K): 144,
    int(WeightType.Q5_K): 176,
    int(WeightType.Q6_K): 210,
    int(WeightType.IQ3_XXS): 98,
    # IQ4_XS has a tiled kernel and repack too, but it is not faster than the dp4a kernel for up to 4 tokens
    # (table lookups dominate) and ~15 % slower for one token, so it stays on the row-major path:
    # int(WeightType.IQ4_XS): 136,
}


def tiled_available() -> bool:
    ns = getattr(torch.ops, "_C_gguf", None)
    return ns is not None and hasattr(ns, "ggml_mul_mat_tiled") and hasattr(ns, "ggml_dequantize_tiled")


def can_tile(weight_type: int, weight: torch.Tensor) -> bool:
    nb = _BLOCK_BYTES.get(int(weight_type))
    return (
        nb is not None
        and weight.dim() == 2
        and weight.dtype == torch.uint8
        and weight.shape[1] % nb == 0
        # tiny matrices (e.g. the fused gated-delta-net gate projection in_proj_ba, 64 rows) gain nothing and
        # made Ornith (Qwen3.5-MoE, MTP 3, CUDA graphs) produce degenerate answers in a few cases (cause not understood)
        and weight.shape[0] >= 128
        and weight.is_cuda
        and tiled_available()
    )


def _tiles(weight: torch.Tensor, nb: int):
    rows, nbytes = weight.shape
    bpr = nbytes // nb
    tiles = (rows + 15) // 16
    w = weight
    if tiles * 16 != rows:
        w = torch.cat([w, torch.zeros(tiles * 16 - rows, nbytes, dtype=w.dtype, device=w.device)])
    return w.view(tiles, 16, bpr, nb).permute(0, 2, 1, 3).contiguous(), rows, bpr


@torch.no_grad()
def repack(weight_type: int, weight: torch.Tensor):
    """[rows, bpr*block] uint8 -> (main, d, rows, cols)."""
    t = int(weight_type)
    x, rows, bpr = _tiles(weight, _BLOCK_BYTES[t])
    if t == int(WeightType.Q3_K):
        # block: hmask[32] qs[64] scales[12] d[2]  ->  hmask | qs | int8 scales[16]; d separate
        sc = x[..., 96:108].to(torch.int32)
        isc = torch.arange(16, device=x.device)
        low = (sc[..., isc % 8] >> (4 * (isc // 8))) & 0xF
        high = ((sc[..., 8 + isc % 4] >> (2 * (isc // 4))) & 3) << 4
        s8 = ((low | high) - 32).to(torch.int8).view(torch.uint8)
        main = torch.cat([x[..., :96], s8], dim=-1)
        d = x[..., 108:110]
    elif t == int(WeightType.Q4_K):
        # block: d[2] dmin[2] scales[12] qs[128]  ->  qs | sc[8] | mn[8]; (d, dmin) separate
        s = x[..., 4:16].to(torch.int32)
        j = torch.arange(8, device=x.device)
        lo = j < 4
        sc = torch.where(lo, s[..., j % 4 + 0] & 63, (s[..., (j % 4) + 8] & 0xF) | ((s[..., j % 4] >> 6) << 4))
        mn = torch.where(lo, s[..., (j % 4) + 4] & 63, (s[..., (j % 4) + 8] >> 4) | ((s[..., (j % 4) + 4] >> 6) << 4))
        main = torch.cat([x[..., 16:144], sc.to(torch.uint8), mn.to(torch.uint8)], dim=-1)
        d = x[..., 0:4]
    elif t == int(WeightType.Q5_K):
        # block: d[2] dmin[2] scales[12] qh[32] qs[128]  ->  qs | qh | sc[8] | mn[8]; (d, dmin) separate
        s = x[..., 4:16].to(torch.int32)
        j = torch.arange(8, device=x.device)
        lo = j < 4
        sc = torch.where(lo, s[..., j % 4 + 0] & 63, (s[..., (j % 4) + 8] & 0xF) | ((s[..., j % 4] >> 6) << 4))
        mn = torch.where(lo, s[..., (j % 4) + 4] & 63, (s[..., (j % 4) + 8] >> 4) | ((s[..., (j % 4) + 4] >> 6) << 4))
        main = torch.cat([x[..., 48:176], x[..., 16:48], sc.to(torch.uint8), mn.to(torch.uint8)], dim=-1)
        d = x[..., 0:4]
    elif t == int(WeightType.Q6_K):
        # block: ql[128] qh[64] scales int8[16] d[2]  ->  same without d (padded to 208 bytes); d separate
        main = x[..., :208]
        d = x[..., 208:210]
    elif t == int(WeightType.IQ3_XXS):
        # block: d[2] qs[64] gas[32]  ->  qs | gas (96 bytes); d separate
        main = x[..., 2:98]
        d = x[..., 0:2]
    elif t == int(WeightType.IQ4_XS):
        # block: d[2] scales_h[2] scales_l[4] qs[128]  ->  qs | int8 sc[8] | pad[8]; d separate
        sh = x[..., 2].to(torch.int32) | (x[..., 3].to(torch.int32) << 8)
        sl = x[..., 4:8].to(torch.int32)
        ib = torch.arange(8, device=x.device)
        low = (sl[..., ib // 2] >> (4 * (ib % 2))) & 0xF
        high = ((sh.unsqueeze(-1) >> (2 * ib)) & 3) << 4
        sc8 = ((low | high) - 32).to(torch.int8).view(torch.uint8)
        pad = torch.zeros(*sc8.shape[:-1], 8, dtype=torch.uint8, device=x.device)
        main = torch.cat([x[..., 8:136], sc8, pad], dim=-1)
        d = x[..., 0:2]
    else:
        raise ValueError(f"unsupported type {t}")
    return main.contiguous().view(-1), d.contiguous().view(-1), rows, bpr * 256
