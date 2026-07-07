"""FlashRT transformer layout primitives."""

from __future__ import annotations

from typing import Optional

import torch

from ._ops import add_op_namespace_prefix, ops


@torch.library.register_fake(add_op_namespace_prefix("fill_neginf_bf16"))
def _fill_neginf_bf16_fake(dst: torch.Tensor) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("add_bias_bf16_"))
def _add_bias_bf16_fake(data: torch.Tensor, bias: torch.Tensor) -> None:
    if data.dim() != 2 or bias.shape != (data.shape[1],):
        raise RuntimeError("add_bias_bf16_ expects data (rows, cols), bias (cols,)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("repeat_interleave_heads_bf16"))
def _repeat_interleave_heads_bf16_fake(src: torch.Tensor, repeat: int, dst: torch.Tensor) -> None:
    if src.dim() != 3 or dst.shape != (src.shape[0], src.shape[1] * repeat, src.shape[2]):
        raise RuntimeError("repeat_interleave_heads_bf16 expects src (seq, heads, dim), dst (seq, heads*repeat, dim)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("text_gather_bf16"))
def _text_gather_bf16_fake(src: torch.Tensor, batch: int, seq: int, dst: torch.Tensor) -> None:
    if src.dim() != 2 or src.shape[0] != batch * seq or dst.shape != (2 * batch, src.shape[1]):
        raise RuntimeError("text_gather_bf16 expects src (batch*seq, dim), dst (2*batch, dim)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("text_scatter_bf16"))
def _text_scatter_bf16_fake(dst: torch.Tensor, src: torch.Tensor, batch: int, seq: int) -> None:
    if dst.dim() != 2 or dst.shape[0] != batch * seq or src.shape != (2 * batch, dst.shape[1]):
        raise RuntimeError("text_scatter_bf16 expects dst (batch*seq, dim), src (2*batch, dim)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("rope_rotate_half_bf16_"))
def _rope_rotate_half_bf16_fake(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> None:
    if x.dim() != 3 or x.shape[2] % 2 != 0 or cos.shape != (x.shape[0], x.shape[2]) or sin.shape != cos.shape:
        raise RuntimeError("rope_rotate_half_bf16_ expects x (seq, heads, even_dim), cos/sin (seq, dim)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("qk_rmsnorm_rope_bf16_"))
def _qk_rmsnorm_rope_bf16_fake(
    qk: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> None:
    if qk.dim() != 3 or weight.shape != (qk.shape[2],) or cos.shape != (qk.shape[0], qk.shape[2]) or sin.shape != cos.shape:
        raise RuntimeError("qk_rmsnorm_rope_bf16_ expects qk (rows, heads, dim), weight (dim,), cos/sin (rows, dim)")
    return None


def fill_neginf_bf16(dst: torch.Tensor) -> torch.Tensor:
    ops.fill_neginf_bf16(dst)
    return dst


def add_bias_bf16_(data: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    ops.add_bias_bf16_(data, bias)
    return data


def repeat_interleave_heads_bf16(
    src: torch.Tensor,
    repeat: int,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty((src.shape[0], src.shape[1] * repeat, src.shape[2]), device=src.device, dtype=src.dtype)
    ops.repeat_interleave_heads_bf16(src, int(repeat), out)
    return out


def text_gather_bf16(
    src: torch.Tensor,
    batch: int,
    seq: int,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty((2 * batch, src.shape[1]), device=src.device, dtype=src.dtype)
    ops.text_gather_bf16(src, int(batch), int(seq), out)
    return out


def text_scatter_bf16(dst: torch.Tensor, src: torch.Tensor, batch: int, seq: int) -> torch.Tensor:
    ops.text_scatter_bf16(dst, src, int(batch), int(seq))
    return dst


def rope_rotate_half_bf16_(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    ops.rope_rotate_half_bf16_(x, cos, sin)
    return x


def qk_rmsnorm_rope_bf16_(
    qk: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    ops.qk_rmsnorm_rope_bf16_(qk, weight, cos, sin, float(eps))
    return qk


__all__ = [
    "fill_neginf_bf16",
    "add_bias_bf16_",
    "repeat_interleave_heads_bf16",
    "text_gather_bf16",
    "text_scatter_bf16",
    "rope_rotate_half_bf16_",
    "qk_rmsnorm_rope_bf16_",
]
