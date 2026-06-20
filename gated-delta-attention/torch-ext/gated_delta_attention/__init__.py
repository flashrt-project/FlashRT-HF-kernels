"""FlashRT Gated Delta attention kernels."""

from __future__ import annotations

from typing import Optional

import torch

from ._ops import add_op_namespace_prefix, ops


def _check_step(q, k, v, g, beta, out) -> None:
    if q.dim() != 3 or q.shape[2] != 128:
        raise RuntimeError("q must have shape (B,H,128)")
    if k.shape != q.shape or v.shape != q.shape:
        raise RuntimeError("k/v must match q")
    if g.shape != q.shape[:2] or beta.shape != g.shape:
        raise RuntimeError("g/beta must have shape (B,H)")
    if out.shape != q.shape:
        raise RuntimeError("out must match q")


def _check_chunk(q, k, v, g, beta, out) -> None:
    if q.dim() != 3 or q.shape[2] != 128:
        raise RuntimeError("q must have shape (S,H,128)")
    if k.shape != q.shape or v.shape != q.shape:
        raise RuntimeError("k/v must match q")
    if g.shape != q.shape[:2] or beta.shape != g.shape:
        raise RuntimeError("g/beta must have shape (S,H)")
    if out.shape != q.shape:
        raise RuntimeError("out must match q")


@torch.library.register_fake(add_op_namespace_prefix("gated_delta_recurrent_bf16"))
def _recurrent_fake(q, k, v, g, beta, state, out, use_qk_l2norm: bool = True) -> None:
    _check_step(q, k, v, g, beta, out)
    if state.shape != (q.shape[0], q.shape[1], 128, 128):
        raise RuntimeError("state must have shape (B,H,128,128)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("gated_delta_recurrent_inout_bf16"))
def _recurrent_inout_fake(q, k, v, g, beta, state_in, state_out, out, use_qk_l2norm: bool = True) -> None:
    _check_step(q, k, v, g, beta, out)
    if state_in.shape != (q.shape[0], q.shape[1], 128, 128) or state_out.shape != state_in.shape:
        raise RuntimeError("state_in/state_out must have shape (B,H,128,128)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("gated_delta_recurrent_f32state_bf16io"))
def _recurrent_f32_fake(q, k, v, g, beta, state_f32, out, use_qk_l2norm: bool = True) -> None:
    _check_step(q, k, v, g, beta, out)
    if state_f32.shape != (q.shape[0], q.shape[1], 128, 128):
        raise RuntimeError("state_f32 must have shape (B,H,128,128)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("gated_delta_chunk_bf16"))
def _chunk_fake(q, k, v, g, beta, state, out, use_qk_l2norm: bool = True) -> None:
    _check_chunk(q, k, v, g, beta, out)
    if state.shape != (q.shape[1], 128, 128):
        raise RuntimeError("state must have shape (H,128,128)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("gated_delta_chunk_smem_bf16"))
def _chunk_smem_fake(q, k, v, g, beta, state, out, use_qk_l2norm: bool = True) -> None:
    _chunk_fake(q, k, v, g, beta, state, out, use_qk_l2norm)
    return None


def gated_delta_recurrent_bf16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    *,
    use_qk_l2norm: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(q)
    ops.gated_delta_recurrent_bf16(q, k, v, g, beta, state, out, bool(use_qk_l2norm))
    return out


def gated_delta_recurrent_inout_bf16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state_in: torch.Tensor,
    *,
    use_qk_l2norm: bool = True,
    state_out: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        out = torch.empty_like(q)
    if state_out is None:
        state_out = torch.empty_like(state_in)
    ops.gated_delta_recurrent_inout_bf16(q, k, v, g, beta, state_in, state_out, out, bool(use_qk_l2norm))
    return out, state_out


def gated_delta_recurrent_f32state_bf16io(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state_f32: torch.Tensor,
    *,
    use_qk_l2norm: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(q)
    ops.gated_delta_recurrent_f32state_bf16io(q, k, v, g, beta, state_f32, out, bool(use_qk_l2norm))
    return out


def gated_delta_chunk_bf16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    *,
    use_qk_l2norm: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(q)
    ops.gated_delta_chunk_bf16(q, k, v, g, beta, state, out, bool(use_qk_l2norm))
    return out


def gated_delta_chunk_smem_bf16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    *,
    use_qk_l2norm: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(q)
    ops.gated_delta_chunk_smem_bf16(q, k, v, g, beta, state, out, bool(use_qk_l2norm))
    return out


__all__ = [
    "gated_delta_recurrent_bf16",
    "gated_delta_recurrent_inout_bf16",
    "gated_delta_recurrent_f32state_bf16io",
    "gated_delta_chunk_bf16",
    "gated_delta_chunk_smem_bf16",
]
