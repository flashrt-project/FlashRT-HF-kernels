"""FlashRT causal Conv1D state kernels."""

from __future__ import annotations

from typing import Optional

import torch

from ._ops import add_op_namespace_prefix, ops


def _empty_bias_like(x: torch.Tensor) -> torch.Tensor:
    return torch.empty((0,), device=x.device, dtype=torch.bfloat16)


def _check_conv_shapes(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor, has_bias: bool, out: torch.Tensor) -> None:
    if x.dim() != 3:
        raise RuntimeError("x must have shape (B,S,C)")
    if w.dim() != 2 or w.shape[0] != x.shape[2] or w.shape[1] < 2 or w.shape[1] > 8:
        raise RuntimeError("w must have shape (C,K), 2 <= K <= 8")
    if has_bias and bias.shape != (x.shape[2],):
        raise RuntimeError("bias must have shape (C,)")
    if out.shape != x.shape:
        raise RuntimeError("out must match x shape")


@torch.library.register_fake(add_op_namespace_prefix("causal_conv1d_bf16"))
def _causal_conv1d_bf16_fake(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    has_bias: bool = True,
    apply_silu: bool = True,
) -> None:
    _check_conv_shapes(x, w, bias, has_bias, out)
    return None


@torch.library.register_fake(add_op_namespace_prefix("causal_conv1d_update_bf16"))
def _causal_conv1d_update_bf16_fake(
    x_new: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor,
    state: torch.Tensor,
    out: torch.Tensor,
    has_bias: bool = True,
    apply_silu: bool = True,
) -> None:
    if x_new.dim() != 2:
        raise RuntimeError("x_new must have shape (B,C)")
    b, c = x_new.shape
    if w.dim() != 2 or w.shape[0] != c or w.shape[1] < 2 or w.shape[1] > 8:
        raise RuntimeError("w must have shape (C,K), 2 <= K <= 8")
    if has_bias and bias.shape != (c,):
        raise RuntimeError("bias must have shape (C,)")
    if state.shape != (b, c, w.shape[1] - 1):
        raise RuntimeError("state must have shape (B,C,K-1)")
    if out.shape != x_new.shape:
        raise RuntimeError("out must match x_new shape")
    return None


@torch.library.register_fake(add_op_namespace_prefix("causal_conv1d_update_inout_bf16"))
def _causal_conv1d_update_inout_bf16_fake(
    x_new: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor,
    state_in: torch.Tensor,
    state_out: torch.Tensor,
    out: torch.Tensor,
    has_bias: bool = True,
    apply_silu: bool = True,
) -> None:
    _causal_conv1d_update_bf16_fake(x_new, w, bias, state_in, out, has_bias, apply_silu)
    if state_out.shape != state_in.shape:
        raise RuntimeError("state_out shape mismatch")
    return None


@torch.library.register_fake(add_op_namespace_prefix("causal_conv1d_update_chunk_bf16"))
def _causal_conv1d_update_chunk_bf16_fake(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor,
    state: torch.Tensor,
    out: torch.Tensor,
    has_bias: bool = True,
    apply_silu: bool = True,
) -> None:
    _check_conv_shapes(x, w, bias, has_bias, out)
    if state.shape != (x.shape[0], x.shape[2], w.shape[1] - 1):
        raise RuntimeError("state must have shape (B,C,K-1)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("causal_conv1d_update_chunk_parallel_bf16"))
def _causal_conv1d_update_chunk_parallel_bf16_fake(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor,
    state: torch.Tensor,
    out: torch.Tensor,
    has_bias: bool = True,
    apply_silu: bool = True,
) -> None:
    _causal_conv1d_update_chunk_bf16_fake(x, w, bias, state, out, has_bias, apply_silu)
    return None


@torch.library.register_fake(add_op_namespace_prefix("causal_conv1d_update_chunk_parallel_gqa_bf16"))
def _causal_conv1d_update_chunk_parallel_gqa_bf16_fake(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor,
    state: torch.Tensor,
    q16: torch.Tensor,
    k16: torch.Tensor,
    v48: torch.Tensor,
    has_bias: bool = True,
    apply_silu: bool = True,
) -> None:
    if x.dim() != 3 or x.shape[2] != 10240:
        raise RuntimeError("GQA split variant expects x shape (B,S,10240)")
    if w.shape != (10240, w.shape[1]) or w.shape[1] < 2 or w.shape[1] > 8:
        raise RuntimeError("w must have shape (10240,K), 2 <= K <= 8")
    b, s, _ = x.shape
    if state.shape != (b, 10240, w.shape[1] - 1):
        raise RuntimeError("state must have shape (B,10240,K-1)")
    if q16.shape != (b, s, 16, 128) or k16.shape != q16.shape or v48.shape != (b, s, 48, 128):
        raise RuntimeError("q16/k16/v48 output shape mismatch")
    if has_bias and bias.shape != (10240,):
        raise RuntimeError("bias must have shape (10240,)")
    return None


def causal_conv1d_bf16(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    apply_silu: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(x)
    has_bias = bias is not None
    if bias is None:
        bias = _empty_bias_like(x)
    ops.causal_conv1d_bf16(x, w, bias, out, bool(has_bias), bool(apply_silu))
    return out


def causal_conv1d_update_bf16(
    x_new: torch.Tensor,
    w: torch.Tensor,
    state: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    apply_silu: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(x_new)
    has_bias = bias is not None
    if bias is None:
        bias = _empty_bias_like(x_new)
    ops.causal_conv1d_update_bf16(x_new, w, bias, state, out, bool(has_bias), bool(apply_silu))
    return out


def causal_conv1d_update_inout_bf16(
    x_new: torch.Tensor,
    w: torch.Tensor,
    state_in: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    apply_silu: bool = True,
    out: Optional[torch.Tensor] = None,
    state_out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        out = torch.empty_like(x_new)
    if state_out is None:
        state_out = torch.empty_like(state_in)
    has_bias = bias is not None
    if bias is None:
        bias = _empty_bias_like(x_new)
    ops.causal_conv1d_update_inout_bf16(
        x_new, w, bias, state_in, state_out, out, bool(has_bias), bool(apply_silu)
    )
    return out, state_out


def causal_conv1d_update_chunk_bf16(
    x: torch.Tensor,
    w: torch.Tensor,
    state: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    apply_silu: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(x)
    has_bias = bias is not None
    if bias is None:
        bias = _empty_bias_like(x)
    ops.causal_conv1d_update_chunk_bf16(x, w, bias, state, out, bool(has_bias), bool(apply_silu))
    return out


def causal_conv1d_update_chunk_parallel_bf16(
    x: torch.Tensor,
    w: torch.Tensor,
    state: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    apply_silu: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(x)
    has_bias = bias is not None
    if bias is None:
        bias = _empty_bias_like(x)
    ops.causal_conv1d_update_chunk_parallel_bf16(x, w, bias, state, out, bool(has_bias), bool(apply_silu))
    return out


def causal_conv1d_update_chunk_parallel_gqa_bf16(
    x: torch.Tensor,
    w: torch.Tensor,
    state: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    apply_silu: bool = True,
    q16: Optional[torch.Tensor] = None,
    k16: Optional[torch.Tensor] = None,
    v48: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, s, _ = x.shape
    if q16 is None:
        q16 = torch.empty((b, s, 16, 128), device=x.device, dtype=torch.bfloat16)
    if k16 is None:
        k16 = torch.empty_like(q16)
    if v48 is None:
        v48 = torch.empty((b, s, 48, 128), device=x.device, dtype=torch.bfloat16)
    has_bias = bias is not None
    if bias is None:
        bias = _empty_bias_like(x)
    ops.causal_conv1d_update_chunk_parallel_gqa_bf16(
        x, w, bias, state, q16, k16, v48, bool(has_bias), bool(apply_silu)
    )
    return q16, k16, v48


__all__ = [
    "causal_conv1d_bf16",
    "causal_conv1d_update_bf16",
    "causal_conv1d_update_inout_bf16",
    "causal_conv1d_update_chunk_bf16",
    "causal_conv1d_update_chunk_parallel_bf16",
    "causal_conv1d_update_chunk_parallel_gqa_bf16",
]
