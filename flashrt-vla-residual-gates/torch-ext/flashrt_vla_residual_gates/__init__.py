"""FlashRT VLA joint residual/gate kernels."""

from __future__ import annotations

import torch

from ._ops import add_op_namespace_prefix, ops


def _check_matrix(x: torch.Tensor, name: str) -> None:
    if x.dim() != 2:
        raise RuntimeError(f"{name} must have shape (rows, dim)")
    if x.shape[1] % 2 != 0:
        raise RuntimeError(f"{name}.shape[1] must be even")


def _check_like(x: torch.Tensor, ref: torch.Tensor, name: str, ref_name: str) -> None:
    if x.shape != ref.shape:
        raise RuntimeError(f"{name} must have the same shape as {ref_name}")


def _check_bias(bias: torch.Tensor, dim: int, name: str) -> None:
    if bias.shape != (dim,):
        raise RuntimeError(f"{name} must have shape (dim,)")


@torch.library.register_fake(add_op_namespace_prefix("joint3_bias_gate_residual_bf16"))
def _joint3_bias_gate_residual_bf16_fake(
    v_residual: torch.Tensor,
    v_x: torch.Tensor,
    v_bias: torch.Tensor,
    v_gate: torch.Tensor,
    v_out: torch.Tensor,
    a_residual: torch.Tensor,
    a_x: torch.Tensor,
    a_bias: torch.Tensor,
    a_gate: torch.Tensor,
    a_out: torch.Tensor,
    u_residual: torch.Tensor,
    u_x: torch.Tensor,
    u_out: torch.Tensor,
) -> None:
    _check_matrix(v_residual, "v_residual")
    _check_like(v_x, v_residual, "v_x", "v_residual")
    _check_like(v_gate, v_residual, "v_gate", "v_residual")
    _check_like(v_out, v_residual, "v_out", "v_residual")
    _check_bias(v_bias, v_residual.shape[1], "v_bias")
    _check_matrix(a_residual, "a_residual")
    _check_like(a_x, a_residual, "a_x", "a_residual")
    _check_like(a_gate, a_residual, "a_gate", "a_residual")
    _check_like(a_out, a_residual, "a_out", "a_residual")
    _check_bias(a_bias, a_residual.shape[1], "a_bias")
    _check_matrix(u_residual, "u_residual")
    _check_like(u_x, u_residual, "u_x", "u_residual")
    _check_like(u_out, u_residual, "u_out", "u_residual")
    return None


@torch.library.register_fake(add_op_namespace_prefix("gate_residual_bf16"))
def _gate_residual_bf16_fake(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    out: torch.Tensor,
) -> None:
    _check_matrix(residual, "residual")
    _check_like(x, residual, "x", "residual")
    _check_like(gate, residual, "gate", "residual")
    _check_like(out, residual, "out", "residual")
    return None


@torch.library.register_fake(add_op_namespace_prefix("bias_residual_bf16"))
def _bias_residual_bf16_fake(
    residual: torch.Tensor,
    x: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
) -> None:
    _check_matrix(residual, "residual")
    _check_like(x, residual, "x", "residual")
    _check_like(out, residual, "out", "residual")
    _check_bias(bias, residual.shape[1], "bias")
    return None


@torch.library.register_fake(add_op_namespace_prefix("joint3_bias_gate_residual_action_nobias_bf16"))
def _joint3_bias_gate_residual_action_nobias_bf16_fake(
    v_residual: torch.Tensor,
    v_x: torch.Tensor,
    v_bias: torch.Tensor,
    v_gate: torch.Tensor,
    v_out: torch.Tensor,
    a_residual: torch.Tensor,
    a_x: torch.Tensor,
    a_gate: torch.Tensor,
    a_out: torch.Tensor,
    u_residual: torch.Tensor,
    u_x: torch.Tensor,
    u_out: torch.Tensor,
) -> None:
    _check_matrix(v_residual, "v_residual")
    _check_like(v_x, v_residual, "v_x", "v_residual")
    _check_like(v_gate, v_residual, "v_gate", "v_residual")
    _check_like(v_out, v_residual, "v_out", "v_residual")
    _check_bias(v_bias, v_residual.shape[1], "v_bias")
    _check_matrix(a_residual, "a_residual")
    _check_like(a_x, a_residual, "a_x", "a_residual")
    _check_like(a_gate, a_residual, "a_gate", "a_residual")
    _check_like(a_out, a_residual, "a_out", "a_residual")
    _check_matrix(u_residual, "u_residual")
    _check_like(u_x, u_residual, "u_x", "u_residual")
    _check_like(u_out, u_residual, "u_out", "u_residual")
    return None


def gate_residual_bf16(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``residual + x * gate`` and store BF16 output.

    ``residual``, ``x``, and ``gate`` must be contiguous BF16 tensors with
    shape ``(rows, dim)``. ``out`` may alias ``residual`` for in-place residual
    stream updates.
    """

    if out is None:
        out = torch.empty_like(residual)
    ops.gate_residual_bf16(residual, x, gate, out)
    return out


def bias_residual_bf16(
    residual: torch.Tensor,
    x: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``residual + x + bias`` and store BF16 output."""

    if out is None:
        out = torch.empty_like(residual)
    ops.bias_residual_bf16(residual, x, bias, out)
    return out


def joint3_bias_gate_residual_bf16(
    v_residual: torch.Tensor,
    v_x: torch.Tensor,
    v_bias: torch.Tensor,
    v_gate: torch.Tensor,
    a_residual: torch.Tensor,
    a_x: torch.Tensor,
    a_bias: torch.Tensor,
    a_gate: torch.Tensor,
    u_residual: torch.Tensor,
    u_x: torch.Tensor,
    v_out: torch.Tensor | None = None,
    a_out: torch.Tensor | None = None,
    u_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute V/A/U residual outputs with video and action bias."""

    if v_out is None:
        v_out = torch.empty_like(v_residual)
    if a_out is None:
        a_out = torch.empty_like(a_residual)
    if u_out is None:
        u_out = torch.empty_like(u_residual)
    ops.joint3_bias_gate_residual_bf16(
        v_residual,
        v_x,
        v_bias,
        v_gate,
        v_out,
        a_residual,
        a_x,
        a_bias,
        a_gate,
        a_out,
        u_residual,
        u_x,
        u_out,
    )
    return v_out, a_out, u_out


def joint3_bias_gate_residual_action_nobias_bf16(
    v_residual: torch.Tensor,
    v_x: torch.Tensor,
    v_bias: torch.Tensor,
    v_gate: torch.Tensor,
    a_residual: torch.Tensor,
    a_x: torch.Tensor,
    a_gate: torch.Tensor,
    u_residual: torch.Tensor,
    u_x: torch.Tensor,
    v_out: torch.Tensor | None = None,
    a_out: torch.Tensor | None = None,
    u_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute V/A/U residual outputs with no action bias."""

    if v_out is None:
        v_out = torch.empty_like(v_residual)
    if a_out is None:
        a_out = torch.empty_like(a_residual)
    if u_out is None:
        u_out = torch.empty_like(u_residual)
    ops.joint3_bias_gate_residual_action_nobias_bf16(
        v_residual,
        v_x,
        v_bias,
        v_gate,
        v_out,
        a_residual,
        a_x,
        a_gate,
        a_out,
        u_residual,
        u_x,
        u_out,
    )
    return v_out, a_out, u_out


__all__ = [
    "bias_residual_bf16",
    "gate_residual_bf16",
    "joint3_bias_gate_residual_bf16",
    "joint3_bias_gate_residual_action_nobias_bf16",
]
