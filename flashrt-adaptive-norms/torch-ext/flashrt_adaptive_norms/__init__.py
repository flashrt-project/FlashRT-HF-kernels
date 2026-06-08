"""FlashRT adaptive norm kernels."""

from __future__ import annotations

import torch

from ._ops import add_op_namespace_prefix, ops


def _check_adaptive_shapes(
    x: torch.Tensor,
    weight: torch.Tensor,
    style: torch.Tensor,
    out: torch.Tensor,
    gate_out: torch.Tensor,
) -> None:
    if x.dim() != 2:
        raise RuntimeError("x must have shape (rows, dim)")
    rows, dim = x.shape
    if dim % 2 != 0:
        raise RuntimeError("x.shape[1] must be even")
    if weight.shape != (dim,):
        raise RuntimeError("weight must have shape (dim,)")
    if style.shape != (rows, 3 * dim):
        raise RuntimeError("style must have shape (rows, 3 * dim)")
    if out.shape != x.shape or gate_out.shape != x.shape:
        raise RuntimeError("out and gate_out must have the same shape as x")


@torch.library.register_fake(add_op_namespace_prefix("ada_rms_norm_style_bf16"))
def _ada_rms_norm_style_bf16_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    style: torch.Tensor,
    eps: float,
    out: torch.Tensor,
    gate_out: torch.Tensor,
) -> None:
    _check_adaptive_shapes(x, weight, style, out, gate_out)
    return None


@torch.library.register_fake(add_op_namespace_prefix("gate_residual_ada_norm_fp8_static_bf16"))
def _gate_residual_ada_norm_fp8_static_bf16_fake(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    style: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
    out: torch.Tensor,
    gate_out: torch.Tensor,
) -> None:
    _check_adaptive_shapes(residual, weight, style, out, gate_out)
    if x.shape != residual.shape or gate.shape != residual.shape:
        raise RuntimeError("x and gate must have the same shape as residual")
    if scale.numel() != 1:
        raise RuntimeError("scale must be a scalar tensor")
    return None


def ada_rms_norm_style_bf16(
    x: torch.Tensor,
    weight: torch.Tensor,
    style: torch.Tensor,
    eps: float = 1e-6,
    out: torch.Tensor | None = None,
    gate_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RMSNorm, style scale/shift, and return the style gate."""

    if out is None:
        out = torch.empty_like(x)
    if gate_out is None:
        gate_out = torch.empty_like(x)
    ops.ada_rms_norm_style_bf16(x, weight, style, float(eps), out, gate_out)
    return out, gate_out


def gate_residual_ada_norm_fp8_static_bf16(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    style: torch.Tensor,
    scale: torch.Tensor,
    eps: float = 1e-6,
    out: torch.Tensor | None = None,
    gate_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update residual in place, apply AdaRMSNorm, and emit static-scale FP8."""

    if out is None:
        out = torch.empty_like(residual, dtype=torch.float8_e4m3fn)
    if gate_out is None:
        gate_out = torch.empty_like(residual)
    ops.gate_residual_ada_norm_fp8_static_bf16(
        residual,
        x,
        gate,
        weight,
        style,
        scale,
        float(eps),
        out,
        gate_out,
    )
    return residual, out, gate_out


__all__ = [
    "ada_rms_norm_style_bf16",
    "gate_residual_ada_norm_fp8_static_bf16",
]
