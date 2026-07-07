"""FlashRT INT8 transformer primitives."""

from __future__ import annotations

from typing import Optional

import torch

from ._ops import add_op_namespace_prefix, ops


@torch.library.register_fake(add_op_namespace_prefix("quantize_int8_static_bf16"))
def _quantize_int8_static_bf16_fake(input: torch.Tensor, scale: torch.Tensor, out: torch.Tensor) -> None:
    if out.shape != input.shape:
        raise RuntimeError("out must match input shape")
    return None


@torch.library.register_fake(add_op_namespace_prefix("quantize_int8_rowwise_bf16"))
def _quantize_int8_rowwise_bf16_fake(input: torch.Tensor, out: torch.Tensor, scales: torch.Tensor) -> None:
    if input.dim() != 2 or out.shape != input.shape or scales.shape != (input.shape[0],):
        raise RuntimeError("quantize_int8_rowwise_bf16 expects input/out (rows, cols), scales (rows,)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("quantize_int8_rowwise_static_bf16"))
def _quantize_int8_rowwise_static_bf16_fake(input: torch.Tensor, scales: torch.Tensor, out: torch.Tensor) -> None:
    return _quantize_int8_rowwise_bf16_fake(input, out, scales)


@torch.library.register_fake(add_op_namespace_prefix("rms_norm_quantize_int8_rowwise_bf16"))
def _rms_norm_quantize_int8_rowwise_bf16_fake(
    x: torch.Tensor, weight: torch.Tensor, eps: float, out: torch.Tensor, scales: torch.Tensor
) -> None:
    return _quantize_int8_rowwise_bf16_fake(x, out, scales)


@torch.library.register_fake(add_op_namespace_prefix("residual_add_rms_norm_quantize_int8_rowwise_bf16"))
def _residual_add_rms_norm_quantize_int8_rowwise_bf16_fake(
    residual: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: torch.Tensor,
    scales: torch.Tensor,
) -> None:
    return _quantize_int8_rowwise_bf16_fake(residual, out, scales)


@torch.library.register_fake(add_op_namespace_prefix("int8_rowwise_linear_bf16"))
def _int8_rowwise_linear_bf16_fake(
    input_i8: torch.Tensor,
    weight_i8: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor,
    variant: int,
) -> None:
    if input_i8.dim() != 2 or weight_i8.dim() != 2 or out.shape != (input_i8.shape[0], weight_i8.shape[0]):
        raise RuntimeError("int8_rowwise_linear_bf16 expects input (M,K), weight (N,K), out (M,N)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("int8_silu_gated_linear_bf16"))
def _int8_silu_gated_linear_bf16_fake(
    input_i8: torch.Tensor,
    up_weight_i8: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    gate: torch.Tensor,
    out: torch.Tensor,
) -> None:
    return _int8_rowwise_linear_bf16_fake(input_i8, up_weight_i8, input_scale, weight_scale, out, 0)


def quantize_int8_static_bf16(
    input: torch.Tensor, scale: torch.Tensor, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input, dtype=torch.int8)
    ops.quantize_int8_static_bf16(input, scale, out)
    return out


def quantize_int8_rowwise_bf16(
    input: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        out = torch.empty_like(input, dtype=torch.int8)
    if scales is None:
        scales = torch.empty((input.shape[0],), device=input.device, dtype=torch.float32)
    ops.quantize_int8_rowwise_bf16(input, out, scales)
    return out, scales


def quantize_int8_rowwise_static_bf16(
    input: torch.Tensor, scales: torch.Tensor, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input, dtype=torch.int8)
    ops.quantize_int8_rowwise_static_bf16(input, scales, out)
    return out


def rms_norm_quantize_int8_rowwise_bf16(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    out: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        out = torch.empty_like(x, dtype=torch.int8)
    if scales is None:
        scales = torch.empty((x.shape[0],), device=x.device, dtype=torch.float32)
    ops.rms_norm_quantize_int8_rowwise_bf16(x, weight, float(eps), out, scales)
    return out, scales


def residual_add_rms_norm_quantize_int8_rowwise_bf16(
    residual: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    out: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        out = torch.empty_like(residual, dtype=torch.int8)
    if scales is None:
        scales = torch.empty((residual.shape[0],), device=residual.device, dtype=torch.float32)
    ops.residual_add_rms_norm_quantize_int8_rowwise_bf16(residual, x, weight, float(eps), out, scales)
    return out, scales


def int8_rowwise_linear_bf16(
    input_i8: torch.Tensor,
    weight_i8: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    variant: int = 0,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty((input_i8.shape[0], weight_i8.shape[0]), device=input_i8.device, dtype=torch.bfloat16)
    ops.int8_rowwise_linear_bf16(input_i8, weight_i8, input_scale, weight_scale, out, int(variant))
    return out


def int8_silu_gated_linear_bf16(
    input_i8: torch.Tensor,
    up_weight_i8: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    gate: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty((input_i8.shape[0], up_weight_i8.shape[0]), device=input_i8.device, dtype=torch.bfloat16)
    ops.int8_silu_gated_linear_bf16(input_i8, up_weight_i8, input_scale, weight_scale, gate, out)
    return out


__all__ = [
    "quantize_int8_static_bf16",
    "quantize_int8_rowwise_bf16",
    "quantize_int8_rowwise_static_bf16",
    "rms_norm_quantize_int8_rowwise_bf16",
    "residual_add_rms_norm_quantize_int8_rowwise_bf16",
    "int8_rowwise_linear_bf16",
    "int8_silu_gated_linear_bf16",
]
