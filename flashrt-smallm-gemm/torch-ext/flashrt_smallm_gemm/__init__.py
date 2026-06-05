"""FlashRT small-M GEMM kernels."""

from __future__ import annotations

from typing import Optional

import torch

from ._ops import add_op_namespace_prefix, ops


@torch.library.register_fake(add_op_namespace_prefix("nvfp4_w4a4_decode_matvec_bf16out"))
def _nvfp4_w4a4_decode_matvec_bf16out_fake(
    a_packed: torch.Tensor,
    b_packed: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    out: torch.Tensor,
    alpha: float = 1.0,
) -> None:
    if b_packed.dim() != 2:
        raise RuntimeError("b_packed must have shape (N, K / 2)")
    if out.shape != (b_packed.shape[0],):
        raise RuntimeError("out shape must be (b_packed.shape[0],)")
    return None


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
