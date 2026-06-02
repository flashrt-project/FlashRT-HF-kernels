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


def qkv_split_norm_rope_bf16(
    packed_qkv: torch.Tensor,
    norm_q_weight: torch.Tensor,
    norm_k_weight: torch.Tensor,
    freqs_re: torch.Tensor,
    freqs_im: torch.Tensor,
    *,
    heads: int,
    head_dim: int,
    seq_len: Optional[int] = None,
    q_out: Optional[torch.Tensor] = None,
    k_out: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split packed QKV, RMS-normalize Q/K, and apply interleaved RoPE.

    ``packed_qkv`` must be contiguous BF16 with shape
    ``(batch, tokens, 3 * heads * head_dim)``. ``norm_q_weight`` and
    ``norm_k_weight`` must have shape ``(heads * head_dim,)``. ``freqs_re`` and
    ``freqs_im`` must be contiguous FP32 tensors with shape
    ``(seq_len_table, head_dim // 2)``. Outputs have shape
    ``(batch, tokens, heads, head_dim)``.
    """

    if seq_len is None:
        seq_len = packed_qkv.shape[1]
    if q_out is None:
        q_out = torch.empty(
            (packed_qkv.shape[0], packed_qkv.shape[1], heads, head_dim),
            device=packed_qkv.device,
            dtype=packed_qkv.dtype,
        )
    if k_out is None:
        k_out = torch.empty_like(q_out)
    ops.qkv_split_norm_rope_bf16(
        packed_qkv,
        norm_q_weight,
        norm_k_weight,
        freqs_re,
        freqs_im,
        q_out,
        k_out,
        heads,
        head_dim,
        seq_len,
        eps,
    )
    return q_out, k_out


__all__ = [
    "q_norm_rope_bf16",
    "k_norm_rope_v_cache_bf16",
    "qkv_split_norm_rope_bf16",
]
