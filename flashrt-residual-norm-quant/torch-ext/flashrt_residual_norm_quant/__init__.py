"""FlashRT residual/RMSNorm/static-FP8 quantization kernels."""

from __future__ import annotations

import torch

from ._ops import add_op_namespace_prefix, ops


def _check_rank2_same_shape(x: torch.Tensor, out: torch.Tensor, out_name: str) -> None:
    if x.dim() != 2:
        raise RuntimeError("x must be rank-2")
    if out.shape != x.shape:
        raise RuntimeError(f"{out_name} must have the same shape as x")


@torch.library.register_fake(add_op_namespace_prefix("rms_norm_bf16"))
def _rms_norm_bf16_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: torch.Tensor,
) -> None:
    _check_rank2_same_shape(x, out, "out")
    if weight.shape != (x.shape[1],):
        raise RuntimeError("weight must have shape (x.shape[1],)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("rms_norm_quant_fp8_static_bf16"))
def _rms_norm_quant_fp8_static_bf16_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
    out: torch.Tensor,
) -> None:
    _check_rank2_same_shape(x, out, "out")
    if weight.shape != (x.shape[1],):
        raise RuntimeError("weight must have shape (x.shape[1],)")
    if scale.numel() != 1:
        raise RuntimeError("scale must contain exactly one value")
    return None


@torch.library.register_fake(
    add_op_namespace_prefix("residual_add_rms_norm_quant_fp8_static_bf16")
)
def _residual_add_rms_norm_quant_fp8_static_bf16_fake(
    residual: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
    out: torch.Tensor,
) -> None:
    if residual.shape != x.shape:
        raise RuntimeError("residual and x must have the same shape")
    _check_rank2_same_shape(x, out, "out")
    if weight.shape != (x.shape[1],):
        raise RuntimeError("weight must have shape (x.shape[1],)")
    if scale.numel() != 1:
        raise RuntimeError("scale must contain exactly one value")
    return None


def rms_norm_bf16(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """BF16 RMSNorm with affine weight."""

    if out is None:
        out = torch.empty_like(x, dtype=torch.bfloat16)
    ops.rms_norm_bf16(x, weight, float(eps), out)
    return out


def rms_norm_quant_fp8_static_bf16(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    eps: float = 1e-6,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """BF16 RMSNorm followed by static-scale FP8 E4M3 quantization."""

    if out is None:
        out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    ops.rms_norm_quant_fp8_static_bf16(x, weight, scale, float(eps), out)
    return out


def residual_add_rms_norm_quant_fp8_static_bf16(
    residual: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    eps: float = 1e-6,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """In-place ``residual += x`` then RMSNorm and static FP8 quantization."""

    if out is None:
        out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    ops.residual_add_rms_norm_quant_fp8_static_bf16(
        residual,
        x,
        weight,
        scale,
        float(eps),
        out,
    )
    return out


__all__ = [
    "residual_add_rms_norm_quant_fp8_static_bf16",
    "rms_norm_bf16",
    "rms_norm_quant_fp8_static_bf16",
]
