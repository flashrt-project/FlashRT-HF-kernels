"""FlashRT NVFP4 layout kernels."""

from __future__ import annotations

from typing import Optional

import torch

from ._ops import ops


def nvfp4_sf_swizzled_bytes(rows: int, D: int) -> int:
    """Return byte count for a CUTLASS Sm1xx NVFP4 swizzled SF buffer."""

    if rows <= 0:
        raise ValueError("rows must be positive")
    if D <= 0 or D % 16 != 0:
        raise ValueError("D must be positive and divisible by 16")
    n_blocks = D // 16
    n_row_super = (rows + 127) // 128
    n_col_super = (n_blocks + 3) // 4
    return n_row_super * n_col_super * 512


def nvfp4_sf_linear_to_swizzled(
    scales: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    is_sfb: bool = False,
) -> torch.Tensor:
    """Convert linear NVFP4 scale bytes to CUTLASS Sm1xx swizzled layout.

    ``scales`` must be contiguous CUDA ``torch.uint8`` with shape
    ``(rows, D / 16)``. If ``out`` is omitted, a flat ``torch.uint8`` output
    tensor with ``nvfp4_sf_swizzled_bytes(rows, D)`` bytes is allocated.
    """

    if scales.dim() != 2:
        raise ValueError("scales must have shape (rows, D / 16)")
    rows = scales.shape[0]
    D = scales.shape[1] * 16
    if out is None:
        out = torch.zeros(
            (nvfp4_sf_swizzled_bytes(rows, D),),
            device=scales.device,
            dtype=torch.uint8,
        )
    ops.nvfp4_sf_linear_to_swizzled(scales, out, D, is_sfb)
    return out


__all__ = [
    "nvfp4_sf_linear_to_swizzled",
    "nvfp4_sf_swizzled_bytes",
]
