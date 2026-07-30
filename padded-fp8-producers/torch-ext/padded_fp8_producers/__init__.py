"""Padding-aware FP8 producers for fixed-shape transformer regions."""

from __future__ import annotations

from typing import Optional

import torch

from ._ops import add_op_namespace_prefix, ops


def _check_norm_shapes(
    input: torch.Tensor,
    weight: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    scale: torch.Tensor,
    output: torch.Tensor,
) -> None:
    if input.dim() != 3 or min(input.shape) <= 0:
        raise RuntimeError("input must have shape (batch, rows, dim)")
    batch, rows, dim = input.shape
    if (
        weight.shape != (dim,)
        or gamma.shape != (batch, dim)
        or beta.shape != (batch, dim)
        or scale.numel() != 1
        or output.dim() != 3
        or output.shape[0] != batch
        or output.shape[1] < rows
        or output.shape[2] != dim
    ):
        raise RuntimeError("invalid norm producer shapes")


def _check_swiglu_shapes(
    gate: torch.Tensor,
    up: torch.Tensor,
    scale: torch.Tensor,
    output: torch.Tensor,
    *,
    merged: bool,
) -> None:
    if gate.dim() != 2 or min(gate.shape) <= 0:
        raise RuntimeError("input must be a non-empty matrix")
    rows = gate.shape[0]
    if merged:
        if gate.shape[1] % 2:
            raise RuntimeError("merged gate_up width must be even")
        dim = gate.shape[1] // 2
    else:
        dim = gate.shape[1]
        if up.shape != gate.shape:
            raise RuntimeError("up must match gate")
    if (
        scale.numel() != 1
        or output.dim() != 2
        or output.shape[0] < rows
        or output.shape[1] != dim
    ):
        raise RuntimeError("output must have shape (padded_rows, dim)")


@torch.library.register_fake(
    add_op_namespace_prefix("adaptive_rms_norm_quant_fp8_padded_bf16")
)
def _adaptive_rms_fake(input, weight, gamma, beta, scale, eps, output) -> None:
    _check_norm_shapes(input, weight, gamma, beta, scale, output)


@torch.library.register_fake(
    add_op_namespace_prefix(
        "residual_add_adaptive_rms_norm_quant_fp8_padded_bf16"
    )
)
def _residual_adaptive_rms_fake(
    residual, input, weight, gamma, beta, scale, eps, residual_out, output
) -> None:
    _check_norm_shapes(input, weight, gamma, beta, scale, output)
    if residual.shape != input.shape or residual_out.shape != input.shape:
        raise RuntimeError("residual and residual_out must match input")


@torch.library.register_fake(
    add_op_namespace_prefix("swiglu_quant_fp8_padded_bf16")
)
def _swiglu_fake(gate, up, scale, output) -> None:
    _check_swiglu_shapes(gate, up, scale, output, merged=False)


@torch.library.register_fake(
    add_op_namespace_prefix("swiglu_merged_quant_fp8_padded_bf16")
)
def _swiglu_merged_bf16_fake(gate_up, scale, output) -> None:
    _check_swiglu_shapes(gate_up, gate_up, scale, output, merged=True)


@torch.library.register_fake(
    add_op_namespace_prefix("swiglu_merged_quant_fp8_padded_fp16")
)
def _swiglu_merged_fp16_fake(gate_up, scale, output) -> None:
    _check_swiglu_shapes(gate_up, gate_up, scale, output, merged=True)


def _allocate_norm(input: torch.Tensor, padded_rows: int) -> torch.Tensor:
    return torch.empty(
        (input.shape[0], padded_rows, input.shape[2]),
        device=input.device,
        dtype=torch.float8_e4m3fn,
    )


def _allocate_swiglu(
    input: torch.Tensor, padded_rows: int, merged: bool
) -> torch.Tensor:
    dim = input.shape[1] // 2 if merged else input.shape[1]
    return torch.empty(
        (padded_rows, dim), device=input.device, dtype=torch.float8_e4m3fn
    )


def adaptive_rms_norm_quant_fp8_padded_bf16(
    input: torch.Tensor,
    weight: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    scale: torch.Tensor,
    eps: float = 1e-6,
    *,
    padded_rows: Optional[int] = None,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if output is None:
        output = _allocate_norm(
            input, input.shape[1] if padded_rows is None else padded_rows
        )
    elif padded_rows is not None and output.shape[1] != padded_rows:
        raise RuntimeError("output padded dimension does not match padded_rows")
    ops.adaptive_rms_norm_quant_fp8_padded_bf16(
        input, weight, gamma, beta, scale, float(eps), output
    )
    return output


def residual_add_adaptive_rms_norm_quant_fp8_padded_bf16(
    residual: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    scale: torch.Tensor,
    eps: float = 1e-6,
    *,
    padded_rows: Optional[int] = None,
    residual_out: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if residual_out is None:
        residual_out = torch.empty_like(input)
    if output is None:
        output = _allocate_norm(
            input, input.shape[1] if padded_rows is None else padded_rows
        )
    elif padded_rows is not None and output.shape[1] != padded_rows:
        raise RuntimeError("output padded dimension does not match padded_rows")
    ops.residual_add_adaptive_rms_norm_quant_fp8_padded_bf16(
        residual,
        input,
        weight,
        gamma,
        beta,
        scale,
        float(eps),
        residual_out,
        output,
    )
    return residual_out, output


def swiglu_quant_fp8_padded_bf16(
    gate: torch.Tensor,
    up: torch.Tensor,
    scale: torch.Tensor,
    *,
    padded_rows: Optional[int] = None,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if output is None:
        output = _allocate_swiglu(
            gate, gate.shape[0] if padded_rows is None else padded_rows, False
        )
    ops.swiglu_quant_fp8_padded_bf16(gate, up, scale, output)
    return output


def swiglu_merged_quant_fp8_padded_bf16(
    gate_up: torch.Tensor,
    scale: torch.Tensor,
    *,
    padded_rows: Optional[int] = None,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if output is None:
        output = _allocate_swiglu(
            gate_up,
            gate_up.shape[0] if padded_rows is None else padded_rows,
            True,
        )
    ops.swiglu_merged_quant_fp8_padded_bf16(gate_up, scale, output)
    return output


def swiglu_merged_quant_fp8_padded_fp16(
    gate_up: torch.Tensor,
    scale: torch.Tensor,
    *,
    padded_rows: Optional[int] = None,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if output is None:
        output = _allocate_swiglu(
            gate_up,
            gate_up.shape[0] if padded_rows is None else padded_rows,
            True,
        )
    ops.swiglu_merged_quant_fp8_padded_fp16(gate_up, scale, output)
    return output


__all__ = [
    "adaptive_rms_norm_quant_fp8_padded_bf16",
    "residual_add_adaptive_rms_norm_quant_fp8_padded_bf16",
    "swiglu_quant_fp8_padded_bf16",
    "swiglu_merged_quant_fp8_padded_bf16",
    "swiglu_merged_quant_fp8_padded_fp16",
]
