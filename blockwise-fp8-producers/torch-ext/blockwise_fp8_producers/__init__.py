"""Blockwise FP8 producers for transformer and world-model regions."""

from __future__ import annotations

from typing import Optional

import torch

from ._ops import add_op_namespace_prefix, ops


def _check_matrix(input: torch.Tensor, output: torch.Tensor, scale: torch.Tensor) -> None:
    if (
        input.dim() != 2
        or input.shape[0] <= 0
        or input.shape[1] <= 0
        or input.shape[1] % 128 != 0
        or output.shape != input.shape
        or scale.shape != (input.shape[0], input.shape[1] // 128)
    ):
        raise RuntimeError(
            "expected input/output (rows, dim) with dim a positive multiple "
            "of 128 and scale (rows, dim / 128)"
        )


@torch.library.register_fake(add_op_namespace_prefix("quantize_fp8_block128_bf16"))
def _quantize_fake(
    input: torch.Tensor, output: torch.Tensor, scale: torch.Tensor
) -> None:
    _check_matrix(input, output, scale)
    return None


@torch.library.register_fake(add_op_namespace_prefix("layer_norm_fp8_block128_bf16"))
def _layer_norm_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    output: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    _check_matrix(input, output, scale)
    if weight.shape != (input.shape[1],) or bias.shape != weight.shape:
        raise RuntimeError("weight and bias must have shape (dim,)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("rms_norm_fp8_block128_bf16"))
def _rms_norm_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    output: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    _check_matrix(input, output, scale)
    if weight.shape != (input.shape[1],):
        raise RuntimeError("weight must have shape (dim,)")
    return None


@torch.library.register_fake(
    add_op_namespace_prefix("residual_add_rms_norm_fp8_block128_bf16")
)
def _residual_rms_norm_fake(
    residual: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual_out: torch.Tensor,
    output: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    _check_matrix(input, output, scale)
    if (
        residual.shape != input.shape
        or residual_out.shape != input.shape
        or weight.shape != (input.shape[1],)
    ):
        raise RuntimeError("residual/output must match input and weight must be (dim,)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("gelu_tanh_fp8_block128_bf16"))
def _gelu_fake(
    input: torch.Tensor, output: torch.Tensor, scale: torch.Tensor
) -> None:
    _check_matrix(input, output, scale)
    return None


@torch.library.register_fake(
    add_op_namespace_prefix("gelu_tanh_bias_fp8_block128_bf16")
)
def _gelu_bias_fake(
    input: torch.Tensor,
    bias: torch.Tensor,
    output: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    _check_matrix(input, output, scale)
    if bias.shape != (input.shape[1],):
        raise RuntimeError("bias must have shape (dim,)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("silu_mul_fp8_block128_bf16"))
def _silu_mul_fake(
    gate: torch.Tensor,
    up: torch.Tensor,
    output: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    _check_matrix(gate, output, scale)
    if up.shape != gate.shape:
        raise RuntimeError("up must match gate")
    return None


@torch.library.register_fake(
    add_op_namespace_prefix("silu_mul_merged_fp8_block128_bf16")
)
def _silu_merged_fake(
    gate_up: torch.Tensor, output: torch.Tensor, scale: torch.Tensor
) -> None:
    if (
        gate_up.dim() != 2
        or gate_up.shape[0] <= 0
        or gate_up.shape[1] <= 0
        or gate_up.shape[1] % 256 != 0
        or output.shape != (gate_up.shape[0], gate_up.shape[1] // 2)
        or scale.shape != (gate_up.shape[0], gate_up.shape[1] // 256)
    ):
        raise RuntimeError(
            "gate_up must be (rows, 2 * dim), dim multiple of 128"
        )
    return None


def _allocate(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(input, dtype=torch.float8_e4m3fn),
        torch.empty(
            (input.shape[0], input.shape[1] // 128),
            device=input.device,
            dtype=torch.float32,
        ),
    )


def quantize_fp8_block128_bf16(
    input: torch.Tensor,
    *,
    output: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if output is None or scale is None:
        allocated_output, allocated_scale = _allocate(input)
        output = allocated_output if output is None else output
        scale = allocated_scale if scale is None else scale
    ops.quantize_fp8_block128_bf16(input, output, scale)
    return output, scale


def layer_norm_fp8_block128_bf16(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-6,
    *,
    output: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if output is None or scale is None:
        allocated_output, allocated_scale = _allocate(input)
        output = allocated_output if output is None else output
        scale = allocated_scale if scale is None else scale
    ops.layer_norm_fp8_block128_bf16(
        input, weight, bias, float(eps), output, scale
    )
    return output, scale


def rms_norm_fp8_block128_bf16(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    output: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if output is None or scale is None:
        allocated_output, allocated_scale = _allocate(input)
        output = allocated_output if output is None else output
        scale = allocated_scale if scale is None else scale
    ops.rms_norm_fp8_block128_bf16(input, weight, float(eps), output, scale)
    return output, scale


def residual_add_rms_norm_fp8_block128_bf16(
    residual: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    residual_out: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if residual_out is None:
        residual_out = torch.empty_like(input)
    if output is None or scale is None:
        allocated_output, allocated_scale = _allocate(input)
        output = allocated_output if output is None else output
        scale = allocated_scale if scale is None else scale
    ops.residual_add_rms_norm_fp8_block128_bf16(
        residual, input, weight, float(eps), residual_out, output, scale
    )
    return residual_out, output, scale


def gelu_tanh_fp8_block128_bf16(
    input: torch.Tensor,
    *,
    output: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if output is None or scale is None:
        allocated_output, allocated_scale = _allocate(input)
        output = allocated_output if output is None else output
        scale = allocated_scale if scale is None else scale
    ops.gelu_tanh_fp8_block128_bf16(input, output, scale)
    return output, scale


def gelu_tanh_bias_fp8_block128_bf16(
    input: torch.Tensor,
    bias: torch.Tensor,
    *,
    output: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if output is None or scale is None:
        allocated_output, allocated_scale = _allocate(input)
        output = allocated_output if output is None else output
        scale = allocated_scale if scale is None else scale
    ops.gelu_tanh_bias_fp8_block128_bf16(input, bias, output, scale)
    return output, scale


def silu_mul_fp8_block128_bf16(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    output: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if output is None or scale is None:
        allocated_output, allocated_scale = _allocate(gate)
        output = allocated_output if output is None else output
        scale = allocated_scale if scale is None else scale
    ops.silu_mul_fp8_block128_bf16(gate, up, output, scale)
    return output, scale


def silu_mul_merged_fp8_block128_bf16(
    gate_up: torch.Tensor,
    *,
    output: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, merged_dim = gate_up.shape
    dim = merged_dim // 2
    if output is None:
        output = torch.empty(
            (rows, dim), device=gate_up.device, dtype=torch.float8_e4m3fn
        )
    if scale is None:
        scale = torch.empty(
            (rows, dim // 128), device=gate_up.device, dtype=torch.float32
        )
    ops.silu_mul_merged_fp8_block128_bf16(gate_up, output, scale)
    return output, scale


__all__ = [
    "quantize_fp8_block128_bf16",
    "layer_norm_fp8_block128_bf16",
    "rms_norm_fp8_block128_bf16",
    "residual_add_rms_norm_fp8_block128_bf16",
    "gelu_tanh_fp8_block128_bf16",
    "gelu_tanh_bias_fp8_block128_bf16",
    "silu_mul_fp8_block128_bf16",
    "silu_mul_merged_fp8_block128_bf16",
]
