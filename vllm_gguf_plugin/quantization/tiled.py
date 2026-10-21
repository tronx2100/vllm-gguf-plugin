# SPDX-License-Identifier: Apache-2.0
"""Tiled weight layouts for the tensor-core small-batch kernels (see csrc/gguf/mmvq_tiled.cuh).

The repack is a lossless permutation done once at load time (the scales are expanded to int8 and the
block scale d is stored in a separate plane). The tiled tensors replace the original GGUF tensor.
"""

import torch
from gguf import GGMLQuantizationType as WeightType

Q3_K_BYTES = 110
Q3T_MAX_TOKENS = 16      # up to this many tokens the tiled mma kernel is used, above it dequantize + GEMM


def q3k_tiled_available() -> bool:
    ns = getattr(torch.ops, "_C_gguf", None)
    return ns is not None and hasattr(ns, "ggml_mul_mat_q3k_tiled") and hasattr(ns, "ggml_dequantize_q3k_tiled")


def can_tile(weight_type: int, weight: torch.Tensor) -> bool:
    return (
        int(weight_type) == int(WeightType.Q3_K)
        and weight.dim() == 2
        and weight.dtype == torch.uint8
        and weight.shape[1] % Q3_K_BYTES == 0
        and weight.is_cuda
        and q3k_tiled_available()
    )


@torch.no_grad()
def repack_q3k(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """[rows, bpr*110] uint8 -> (main, d, rows, cols): main = tiles x bpr x 16 x 112 bytes (hmask | qs | int8 scales),
    d = tiles x bpr x 16 half."""
    rows, nbytes = weight.shape
    bpr = nbytes // Q3_K_BYTES
    tiles = (rows + 15) // 16
    w = weight
    if tiles * 16 != rows:
        w = torch.cat([w, torch.zeros(tiles * 16 - rows, nbytes, dtype=w.dtype, device=w.device)])
    x = w.view(tiles, 16, bpr, Q3_K_BYTES).permute(0, 2, 1, 3).contiguous()   # [tiles, bpr, 16, 110]
    hq = x[..., :96]
    sc = x[..., 96:108].to(torch.int32)
    isc = torch.arange(16, device=w.device)
    low = (sc[..., isc % 8] >> (4 * (isc // 8))) & 0xF
    high = ((sc[..., 8 + isc % 4] >> (2 * (isc // 4))) & 3) << 4
    s8 = ((low | high) - 32).to(torch.int8).view(torch.uint8)                   # [tiles, bpr, 16, 16]
    main = torch.cat([hq, s8], dim=-1).contiguous().view(-1)
    d = x[..., 108:110].contiguous().view(-1)
    return main, d, rows, bpr * 256
