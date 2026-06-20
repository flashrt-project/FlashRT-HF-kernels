"""FlashRT VL transformer primitive kernels."""

from __future__ import annotations

from typing import Optional

import torch

from ._ops import add_op_namespace_prefix, ops


def _check_decode_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out: torch.Tensor,
    name: str,
) -> None:
    if x.dim() != 2 or x.shape[1] != 128:
        raise RuntimeError(f"{name} must have shape (heads, 128)")
    if weight.shape != (128,):
        raise RuntimeError("norm weight must have shape (128,)")
    if cos.shape != (64,) or sin.shape != (64,):
        raise RuntimeError("cos and sin must have shape (64,)")
    if out.shape != x.shape:
        raise RuntimeError("out must have the same shape as input")


@torch.library.register_fake(add_op_namespace_prefix("qwen3_q_norm_rope_qstage_bf16"))
def _qwen3_q_norm_rope_qstage_bf16_fake(
    q_pre: torch.Tensor,
    q_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    q_out: torch.Tensor,
) -> None:
    _check_decode_rope(q_pre, q_norm_weight, cos, sin, q_out, "q_pre")
    return None


@torch.library.register_fake(add_op_namespace_prefix("qwen3_k_norm_rope_kvwrite_bf16"))
def _qwen3_k_norm_rope_kvwrite_bf16_fake(
    k_pre: torch.Tensor,
    v_pre: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    k_cache_dst: torch.Tensor,
    v_cache_dst: torch.Tensor,
) -> None:
    _check_decode_rope(k_pre, k_norm_weight, cos, sin, k_cache_dst, "k_pre")
    if v_pre.shape != k_pre.shape or v_cache_dst.shape != k_pre.shape:
        raise RuntimeError("v_pre and v_cache_dst must have shape (n_kv_heads, 128)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("qwen3_k_norm_rope_kvwrite_devpos_bf16"))
def _qwen3_k_norm_rope_kvwrite_devpos_bf16_fake(
    k_pre: torch.Tensor,
    v_pre: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cur_pos: torch.Tensor,
    eps: float,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> None:
    if k_pre.dim() != 2 or k_pre.shape[1] != 128:
        raise RuntimeError("k_pre must have shape (n_kv_heads, 128)")
    n_kv = k_pre.shape[0]
    if v_pre.shape != k_pre.shape:
        raise RuntimeError("v_pre must have shape (n_kv_heads, 128)")
    if k_norm_weight.shape != (128,):
        raise RuntimeError("k_norm_weight must have shape (128,)")
    if cos.shape != (64,) or sin.shape != (64,):
        raise RuntimeError("cos and sin must have shape (64,)")
    if cur_pos.numel() != 1:
        raise RuntimeError("cur_pos must have one int32 element")
    if k_cache.dim() != 3 or k_cache.shape[1:] != (n_kv, 128):
        raise RuntimeError("k_cache must have shape (max_seq_len, n_kv_heads, 128)")
    if v_cache.shape != k_cache.shape:
        raise RuntimeError("v_cache must have the same shape as k_cache")
    return None


@torch.library.register_fake(add_op_namespace_prefix("avg_pool_vision_tokens_bf16"))
def _avg_pool_vision_tokens_bf16_fake(
    x: torch.Tensor,
    nv: int,
    h: int,
    w: int,
    pool_factor: int,
    out: torch.Tensor,
) -> None:
    if x.dim() != 2:
        raise RuntimeError("x must have shape (nv * h * w, dim)")
    dim = x.shape[1]
    if x.shape[0] != nv * h * w:
        raise RuntimeError("x.shape[0] must equal nv * h * w")
    if h % pool_factor != 0 or w % pool_factor != 0:
        raise RuntimeError("h and w must be divisible by pool_factor")
    expected = (nv * (h // pool_factor) * (w // pool_factor), dim)
    if out.shape != expected:
        raise RuntimeError("out has the wrong pooled shape")
    return None


def qwen3_q_norm_rope_qstage_bf16(
    q_pre: torch.Tensor,
    q_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply Q RMSNorm + full RoPE and write decode staging output.

    Inputs are contiguous BF16 tensors. ``q_pre`` has shape
    ``(n_q_heads, 128)``. ``q_norm_weight`` has shape ``(128,)``.
    ``cos`` and ``sin`` have shape ``(64,)``.
    """

    if out is None:
        out = torch.empty_like(q_pre)
    ops.qwen3_q_norm_rope_qstage_bf16(q_pre, q_norm_weight, cos, sin, float(eps), out)
    return out


def qwen3_k_norm_rope_kvwrite_bf16(
    k_pre: torch.Tensor,
    v_pre: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    eps: float = 1e-6,
    k_cache_dst: Optional[torch.Tensor] = None,
    v_cache_dst: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply K RMSNorm + full RoPE and copy V into the current KV slot."""

    if k_cache_dst is None:
        k_cache_dst = torch.empty_like(k_pre)
    if v_cache_dst is None:
        v_cache_dst = torch.empty_like(v_pre)
    ops.qwen3_k_norm_rope_kvwrite_bf16(
        k_pre,
        v_pre,
        k_norm_weight,
        cos,
        sin,
        float(eps),
        k_cache_dst,
        v_cache_dst,
    )
    return k_cache_dst, v_cache_dst


def qwen3_k_norm_rope_kvwrite_devpos_bf16(
    k_pre: torch.Tensor,
    v_pre: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cur_pos: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Device-position KV write for CUDA Graph decode loops."""

    ops.qwen3_k_norm_rope_kvwrite_devpos_bf16(
        k_pre,
        v_pre,
        k_norm_weight,
        cos,
        sin,
        cur_pos,
        float(eps),
        k_cache,
        v_cache,
    )
    return k_cache, v_cache


def avg_pool_vision_tokens_bf16(
    x: torch.Tensor,
    nv: int,
    h: int,
    w: int,
    pool_factor: int,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Average-pool flattened vision tokens in the spatial grid.

    ``x`` is ``(nv * h * w, dim)`` BF16. The result is
    ``(nv * h / pool_factor * w / pool_factor, dim)`` BF16.
    """

    if x.dim() != 2:
        raise ValueError("x must have shape (nv * h * w, dim)")
    if h % pool_factor != 0 or w % pool_factor != 0:
        raise ValueError("h and w must be divisible by pool_factor")
    rows = nv * (h // pool_factor) * (w // pool_factor)
    if out is None:
        out = torch.empty((rows, x.shape[1]), device=x.device, dtype=x.dtype)
    ops.avg_pool_vision_tokens_bf16(x, int(nv), int(h), int(w), int(pool_factor), out)
    return out


__all__ = [
    "avg_pool_vision_tokens_bf16",
    "qwen3_k_norm_rope_kvwrite_bf16",
    "qwen3_k_norm_rope_kvwrite_devpos_bf16",
    "qwen3_q_norm_rope_qstage_bf16",
]
