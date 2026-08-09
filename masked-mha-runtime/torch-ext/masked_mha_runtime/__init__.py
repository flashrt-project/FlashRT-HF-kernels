"""Allocation-free masked MHA runtime operators."""

from __future__ import annotations

import math
from typing import Optional

import torch

from ._ops import add_op_namespace_prefix, ops


@torch.library.register_fake(add_op_namespace_prefix("forward_static"))
def _forward_static_fake(q, k, v, logits, out, scale: float) -> None:
    del scale
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise RuntimeError("q/k/v must have shape (sequence, heads, head_dim)")
    if out.shape != q.shape:
        raise RuntimeError("out must match q")
    if logits.dim() != 3 or logits.shape[:2] != (q.shape[1], q.shape[0]):
        raise RuntimeError("logits must have shape (heads, sequence_q, stride)")
    if logits.shape[2] < k.shape[0]:
        raise RuntimeError("logits stride must cover sequence_kv")


@torch.library.register_fake(add_op_namespace_prefix("attention_mha_fp16_masked"))
def _attention_mha_fp16_masked_fake(q, k, v, logits, out, scale: float) -> None:
    _forward_static_fake(q, k, v, logits, out, scale)


@torch.library.register_fake(add_op_namespace_prefix("attention_mha_bf16_masked"))
def _attention_mha_bf16_masked_fake(
    q, k, v, logits, out, scale: float, qkv_token_stride: int
) -> None:
    _forward_static_fake(q, k, v, logits, out, scale)
    if qkv_token_stride != q.stride(0):
        raise RuntimeError("qkv_token_stride must match q.stride(0)")


@torch.library.register_fake(add_op_namespace_prefix("forward_seqused_static"))
def _forward_seqused_static_fake(q, k, v, valid_k, logits, out, scale: float) -> None:
    del scale
    if q.dim() != 3 or k.dim() != 2 or v.dim() != 2:
        raise RuntimeError("q must be (sequence, heads, head_dim); k/v must be (max_sequence, head_dim)")
    if valid_k.numel() != 1 or valid_k.dtype != torch.int32:
        raise RuntimeError("valid_k must be an int32 scalar tensor")
    if out.shape != q.shape or logits.shape != (q.shape[0] * q.shape[1], k.shape[0]):
        raise RuntimeError("caller-owned output shapes do not match q/k")


def allocate_workspace(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Allocate padded logits scratch once, outside the hot path."""
    stride = (k.shape[0] + 7) // 8 * 8
    return torch.empty(
        (q.shape[1], q.shape[0], stride), device=q.device, dtype=q.dtype
    )


def forward_static(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    logits: torch.Tensor,
    out: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Run MHA without pre-filling padded logits; all buffers are caller-owned."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    ops.forward_static(q, k, v, logits, out, float(scale))
    return out


def attention_mha_fp16_masked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    logits: torch.Tensor,
    out: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Run the deterministic valid-column FP16 masked MHA kernel."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    ops.attention_mha_fp16_masked(q, k, v, logits, out, float(scale))
    return out


def attention_mha_bf16_masked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    logits: torch.Tensor,
    out: torch.Tensor,
    qkv_token_stride: Optional[int] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Run BF16 masked MHA directly from contiguous or fused-QKV views."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    if qkv_token_stride is None:
        qkv_token_stride = q.stride(0)
    ops.attention_mha_bf16_masked(
        q, k, v, logits, out, float(scale), int(qkv_token_stride)
    )
    return out


def forward(q, k, v, *, scale: Optional[float] = None):
    """Convenience allocation wrapper; use ``forward_static`` in hot paths."""
    logits = allocate_workspace(q, k)
    out = torch.empty_like(q, memory_format=torch.contiguous_format)
    return forward_static(q, k, v, logits=logits, out=out, scale=scale)


def forward_seqused_static(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_k: torch.Tensor,
    *,
    logits: torch.Tensor,
    out: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Run FP16 shared-KV attention with a graph-resident valid-key count."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    ops.forward_seqused_static(q, k, v, valid_k, logits, out, float(scale))
    return out


__all__ = [
    "attention_mha_bf16_masked",
    "attention_mha_fp16_masked",
    "allocate_workspace",
    "forward",
    "forward_static",
    "forward_seqused_static",
]
