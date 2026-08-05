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


def forward(q, k, v, *, scale: Optional[float] = None):
    """Convenience allocation wrapper; use ``forward_static`` in hot paths."""
    logits = allocate_workspace(q, k)
    out = torch.empty_like(q, memory_format=torch.contiguous_format)
    return forward_static(q, k, v, logits=logits, out=out, scale=scale)


__all__ = ["allocate_workspace", "forward", "forward_static"]
