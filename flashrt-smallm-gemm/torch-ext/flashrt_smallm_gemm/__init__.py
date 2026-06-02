"""FlashRT small-M GEMM kernels."""

from __future__ import annotations

from typing import Optional

import torch

from ._ops import ops


def nvfp4_w4a4_decode_matvec_bf16out(
    a_packed: torch.Tensor,
    b_packed: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    *,
    alpha: float = 1.0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute an SM120 NVFP4 W4A4 M=1 matvec with BF16 output.

    ``a_packed`` contains one packed activation row with shape ``(K / 2,)`` or
    ``(1, K / 2)``. ``b_packed`` is row-major with shape ``(N, K / 2)``. ``sfa``
    and ``sfb`` are CUTLASS Sm1xx swizzled UE4M3 scale-factor byte buffers.
    The current kernel supports ``K in {4096, 12288}``.
    """

    if b_packed.dim() != 2:
        raise ValueError("b_packed must have shape (N, K / 2)")
    if out is None:
        out = torch.empty((b_packed.shape[0],), device=b_packed.device, dtype=torch.bfloat16)
    ops.nvfp4_w4a4_decode_matvec_bf16out(
        a_packed,
        b_packed,
        sfa,
        sfb,
        out,
        float(alpha),
    )
    return out


__all__ = [
    "nvfp4_w4a4_decode_matvec_bf16out",
]
