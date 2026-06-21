"""FlashRT BF16 decode GEMV kernels."""

from __future__ import annotations

from typing import Optional

import torch

from ._ops import add_op_namespace_prefix, ops


@torch.library.register_fake(add_op_namespace_prefix("bf16_decode_gemv_bf16"))
def _bf16_decode_gemv_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    alpha: float,
    variant: int,
    out: torch.Tensor,
) -> None:
    k = x.shape[0] if x.dim() == 1 else x.shape[1]
    if weight.dim() != 2 or weight.shape[1] != k or out.shape != (weight.shape[0],):
        raise RuntimeError("expected x (K,) or (1,K), weight (N,K), out (N,)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("bf16_decode_gemv_unrolled_bf16"))
def _bf16_decode_gemv_unrolled_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
) -> None:
    k = x.shape[0] if x.dim() == 1 else x.shape[1]
    if weight.dim() != 2 or weight.shape[1] != k or out.shape != (weight.shape[0],):
        raise RuntimeError("expected x (K,) or (1,K), weight (N,K), out (N,)")
    return None


def bf16_decode_gemv_bf16(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    alpha: float = 1.0,
    variant: int = 0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute BF16 M=1 decode GEMV: ``out = x @ weight.T * alpha``."""

    if out is None:
        out = torch.empty((weight.shape[0],), device=x.device, dtype=torch.bfloat16)
    ops.bf16_decode_gemv_bf16(x, weight, float(alpha), int(variant), out)
    return out


def bf16_decode_gemv_unrolled_bf16(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute BF16 M=1 GEMV with the unrolled memory-level-parallel kernel."""

    if out is None:
        out = torch.empty((weight.shape[0],), device=x.device, dtype=torch.bfloat16)
    ops.bf16_decode_gemv_unrolled_bf16(x, weight, out)
    return out


__all__ = [
    "bf16_decode_gemv_bf16",
    "bf16_decode_gemv_unrolled_bf16",
]
