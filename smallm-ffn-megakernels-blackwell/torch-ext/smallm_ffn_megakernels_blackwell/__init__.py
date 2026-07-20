"""Static-buffer small-M FP8 FFN regions for Blackwell."""

from __future__ import annotations
import torch
from ._ops import add_op_namespace_prefix, ops


@torch.library.custom_op(
    add_op_namespace_prefix("_gated_region"), mutates_args=(), device_types="cuda"
)
def _gated_region(
    x: torch.Tensor,
    uw: torch.Tensor,
    ub: torch.Tensor,
    dinv: torch.Tensor,
    dw: torch.Tensor,
    db: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
    up_alpha: float,
    down_alpha: float,
    hidden_scale: float,
) -> torch.Tensor:
    out = torch.empty_like(residual)
    scratch = torch.empty(
        (x.shape[0], 4096), device=x.device, dtype=torch.float8_e4m3fn
    )
    ops.fp8_gelu_ffn_gated_residual_bf16_out(
        x,
        uw,
        ub,
        dinv,
        dw,
        db,
        gate,
        residual,
        up_alpha,
        down_alpha,
        hidden_scale,
        out,
        scratch,
    )
    return out


@torch.library.register_fake(add_op_namespace_prefix("_gated_region"))
def _gated_fake(
    x, uw, ub, dinv, dw, db, gate, residual, up_alpha, down_alpha, hidden_scale
):
    return torch.empty_like(residual)


@torch.library.custom_op(
    add_op_namespace_prefix("_residual_region"), mutates_args=(), device_types="cuda"
)
def _residual_region(
    x: torch.Tensor,
    uinv: torch.Tensor,
    uw: torch.Tensor,
    ub: torch.Tensor,
    dinv: torch.Tensor,
    dw: torch.Tensor,
    db: torch.Tensor,
    residual: torch.Tensor,
    up_alpha: float,
    down_alpha: float,
    input_scale: float,
    hidden_scale: float,
    split_stage: bool,
) -> torch.Tensor:
    out = torch.empty_like(residual)
    xs = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    hs = torch.empty((x.shape[0], 2048), device=x.device, dtype=torch.float8_e4m3fn)
    barrier = torch.zeros(2, device=x.device, dtype=torch.uint32)
    ops.fp8_gelu_ffn_residual_bf16_out(
        x,
        uinv,
        uw,
        ub,
        dinv,
        dw,
        db,
        residual,
        up_alpha,
        down_alpha,
        input_scale,
        hidden_scale,
        split_stage,
        out,
        xs,
        hs,
        barrier,
    )
    return out


@torch.library.register_fake(add_op_namespace_prefix("_residual_region"))
def _residual_fake(
    x,
    uinv,
    uw,
    ub,
    dinv,
    dw,
    db,
    residual,
    up_alpha,
    down_alpha,
    input_scale,
    hidden_scale,
    split_stage,
):
    return torch.empty_like(residual)


def fp8_gelu_ffn_gated_residual_bf16_static(
    x,
    up_weight,
    up_bias,
    down_inverse_scale,
    down_weight,
    down_bias,
    gate,
    residual,
    *,
    up_alpha,
    down_alpha,
    hidden_scale,
    out=None,
    hidden_scratch=None,
):
    """Fused FP8 GELU FFN + gate + residual, fixed 1024/4096 dimensions."""
    if out is None and hidden_scratch is None:
        return _gated_region(
            x,
            up_weight,
            up_bias,
            down_inverse_scale,
            down_weight,
            down_bias,
            gate,
            residual,
            float(up_alpha),
            float(down_alpha),
            float(hidden_scale),
        )
    out = torch.empty_like(residual) if out is None else out
    hidden_scratch = (
        torch.empty((x.shape[0], 4096), device=x.device, dtype=torch.float8_e4m3fn)
        if hidden_scratch is None
        else hidden_scratch
    )
    ops.fp8_gelu_ffn_gated_residual_bf16_out(
        x,
        up_weight,
        up_bias,
        down_inverse_scale,
        down_weight,
        down_bias,
        gate,
        residual,
        float(up_alpha),
        float(down_alpha),
        float(hidden_scale),
        out,
        hidden_scratch,
    )
    return out


def fp8_gelu_ffn_residual_bf16_static(
    x,
    up_inverse_scale,
    up_weight,
    up_bias,
    down_inverse_scale,
    down_weight,
    down_bias,
    residual,
    *,
    up_alpha,
    down_alpha,
    input_scale,
    hidden_scale,
    split_stage=False,
    out=None,
    input_scratch=None,
    hidden_scratch=None,
    barrier=None,
):
    """Fused BF16-input FP8 GELU FFN + residual, fixed 512/2048 dimensions."""
    if (
        out is None
        and input_scratch is None
        and hidden_scratch is None
        and barrier is None
    ):
        return _residual_region(
            x,
            up_inverse_scale,
            up_weight,
            up_bias,
            down_inverse_scale,
            down_weight,
            down_bias,
            residual,
            float(up_alpha),
            float(down_alpha),
            float(input_scale),
            float(hidden_scale),
            bool(split_stage),
        )
    m = x.shape[0]
    out = torch.empty_like(residual) if out is None else out
    input_scratch = (
        torch.empty_like(x, dtype=torch.float8_e4m3fn)
        if input_scratch is None
        else input_scratch
    )
    hidden_scratch = (
        torch.empty((m, 2048), device=x.device, dtype=torch.float8_e4m3fn)
        if hidden_scratch is None
        else hidden_scratch
    )
    barrier = (
        torch.zeros(2, device=x.device, dtype=torch.uint32)
        if barrier is None
        else barrier
    )
    ops.fp8_gelu_ffn_residual_bf16_out(
        x,
        up_inverse_scale,
        up_weight,
        up_bias,
        down_inverse_scale,
        down_weight,
        down_bias,
        residual,
        float(up_alpha),
        float(down_alpha),
        float(input_scale),
        float(hidden_scale),
        bool(split_stage),
        out,
        input_scratch,
        hidden_scratch,
        barrier,
    )
    return out


__all__ = [
    "fp8_gelu_ffn_gated_residual_bf16_static",
    "fp8_gelu_ffn_residual_bf16_static",
]
