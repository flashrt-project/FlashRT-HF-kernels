"""FlashRT VLA, vision, video, and diffusion kernels."""

from __future__ import annotations

from typing import Optional

import torch

from ._ops import ops


def q_norm_rope_bf16(
    q: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute per-row RMSNorm plus rotate-half RoPE for BF16 Q heads.

    ``q`` must be contiguous BF16 with shape ``(..., 128)``. ``weight`` must
    have shape ``(128,)``. ``cos`` and ``sin`` must have shape ``(64,)`` for
    the current decode position. If ``out`` is omitted, a BF16 tensor with the
    same shape as ``q`` is allocated.
    """

    if out is None:
        out = torch.empty_like(q)
    ops.q_norm_rope_bf16(q, weight, cos, sin, out, eps)
    return out


def k_norm_rope_v_cache_bf16(
    k: torch.Tensor,
    v: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_out: Optional[torch.Tensor] = None,
    v_out: Optional[torch.Tensor] = None,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute K RMSNorm plus RoPE and copy V into cache-shaped outputs.

    ``k`` and ``v`` must be contiguous BF16 tensors with shape ``(..., 128)``.
    ``weight`` must have shape ``(128,)``. ``cos`` and ``sin`` must have shape
    ``(64,)`` for the current decode position. If outputs are omitted, BF16
    tensors with the same shape as ``k`` are allocated.
    """

    if k_out is None:
        k_out = torch.empty_like(k)
    if v_out is None:
        v_out = torch.empty_like(v)
    ops.k_norm_rope_v_cache_bf16(k, v, weight, cos, sin, k_out, v_out, eps)
    return k_out, v_out


__all__ = [
    "q_norm_rope_bf16",
    "k_norm_rope_v_cache_bf16",
]
