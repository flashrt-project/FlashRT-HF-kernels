"""FlashRT grouped MoE GEMV kernels."""

from __future__ import annotations

from typing import Optional

import torch

from ._ops import add_op_namespace_prefix, ops


@torch.library.register_fake(add_op_namespace_prefix("w4a16_decode_gemv_bf16"))
def _w4a16_decode_gemv_fake(
    x_bf16: torch.Tensor,
    weight_packed: torch.Tensor,
    sfb: torch.Tensor,
    alpha: float,
    out: torch.Tensor,
) -> None:
    k = x_bf16.shape[0] if x_bf16.dim() == 1 else x_bf16.shape[1]
    if weight_packed.dim() != 2 or weight_packed.shape[1] != k // 2 or out.shape != (weight_packed.shape[0],):
        raise RuntimeError("expected x (K,) or (1,K), weight_packed (N,K/2), out (N,)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("grouped_w4a16_gemv_bf16"))
def _grouped_w4a16_gemv_fake(
    activations: torch.Tensor,
    weight_stack: torch.Tensor,
    sfb_stack: torch.Tensor,
    alpha_stack: torch.Tensor,
    expert_idx: torch.Tensor,
    w_stride: int,
    sfb_stride: int,
    out: torch.Tensor,
) -> None:
    if activations.dim() != 2 or out.dim() != 2 or out.shape[0] != activations.shape[0]:
        raise RuntimeError("expected activations (slots,K), out (slots,N)")
    if expert_idx.shape != (activations.shape[0],):
        raise RuntimeError("expert_idx must have shape (slots,)")
    return None


def w4a16_decode_gemv_bf16(
    x_bf16: torch.Tensor,
    weight_packed: torch.Tensor,
    sfb: torch.Tensor,
    *,
    alpha: float = 1.0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty((weight_packed.shape[0],), device=x_bf16.device, dtype=torch.bfloat16)
    ops.w4a16_decode_gemv_bf16(x_bf16, weight_packed, sfb, float(alpha), out)
    return out


def grouped_w4a16_gemv_bf16(
    activations: torch.Tensor,
    weight_stack: torch.Tensor,
    sfb_stack: torch.Tensor,
    alpha_stack: torch.Tensor,
    expert_idx: torch.Tensor,
    *,
    n: int,
    w_stride: Optional[int] = None,
    sfb_stride: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run one W4A16 GEMV per routed slot.

    `weight_stack` is a flat expert stack. `w_stride` and `sfb_stride` are byte
    strides between experts; by default `w_stride = n * K / 2`.
    """

    k = activations.shape[1]
    if out is None:
        out = torch.empty((activations.shape[0], int(n)), device=activations.device, dtype=torch.bfloat16)
    if w_stride is None:
        w_stride = int(n) * k // 2
    if sfb_stride is None:
        raise RuntimeError("sfb_stride must be provided because swizzled SF size is layout-dependent")
    ops.grouped_w4a16_gemv_bf16(
        activations,
        weight_stack,
        sfb_stack,
        alpha_stack,
        expert_idx,
        int(w_stride),
        int(sfb_stride),
        out,
    )
    return out


__all__ = [
    "grouped_w4a16_gemv_bf16",
    "w4a16_decode_gemv_bf16",
]
