"""FlashRT linear-attention helper kernels."""

from __future__ import annotations

from typing import Optional

import torch

from ._ops import add_op_namespace_prefix, ops


@torch.library.register_fake(add_op_namespace_prefix("bf16_matvec"))
def _bf16_matvec_fake(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor) -> None:
    if x.dim() != 1 or w.dim() != 2 or w.shape[1] != x.shape[0] or out.shape != (w.shape[0],):
        raise RuntimeError("bf16_matvec expects x (K,), w (N,K), out (N,)")
    if w.shape[0] < 256:
        raise RuntimeError("bf16_matvec supports N >= 256")
    return None


@torch.library.register_fake(add_op_namespace_prefix("bf16_smallm_matmul"))
def _bf16_smallm_matmul_fake(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor) -> None:
    if x.dim() != 2 or w.dim() != 2 or w.shape[1] != x.shape[1] or out.shape != (x.shape[0], w.shape[0]):
        raise RuntimeError("bf16_smallm_matmul expects x (M,K), w (N,K), out (M,N)")
    if w.shape != (96, 5120):
        raise RuntimeError("bf16_smallm_matmul supports tuned AB96 shape N=96,K=5120")
    if x.shape[0] < 2 or x.shape[0] > 4:
        raise RuntimeError("bf16_smallm_matmul supports 2 <= M <= 4")
    return None


@torch.library.register_fake(add_op_namespace_prefix("split_qkv_broadcast_bf16"))
def _split_qkv_broadcast_fake(
    packed: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    v_heads: int,
    head_dim: int,
) -> None:
    rows = packed.shape[0]
    if packed.shape != (rows, (q_heads + kv_heads + v_heads) * head_dim):
        raise RuntimeError("packed shape mismatch")
    if q.shape != (rows, v_heads, head_dim) or k.shape != (rows, v_heads, head_dim) or v.shape != (rows, v_heads, head_dim):
        raise RuntimeError("q/k/v output shape mismatch")
    return None


@torch.library.register_fake(add_op_namespace_prefix("partial_rope_qk_bf16"))
def _partial_rope_qk_fake(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    rope_dim: int,
) -> None:
    if q_in.dim() != 3 or k_in.dim() != 3 or q_in.shape[0] != k_in.shape[0] or q_in.shape[2] != k_in.shape[2]:
        raise RuntimeError("q_in/k_in shape mismatch")
    rows, _, head_dim = q_in.shape
    if rope_dim <= 0 or rope_dim > head_dim or rope_dim % 2 != 0:
        raise RuntimeError("invalid rope_dim")
    if cos.shape != (rows, rope_dim) or sin.shape != (rows, rope_dim):
        raise RuntimeError("cos/sin shape mismatch")
    if q_out.shape != q_in.shape or k_out.shape != k_in.shape:
        raise RuntimeError("q_out/k_out shape mismatch")
    return None


@torch.library.register_fake(add_op_namespace_prefix("gated_delta_prepare_bf16"))
def _gated_delta_prepare_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    neg_exp_a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    g_out: torch.Tensor,
    beta_out: torch.Tensor,
    a_stride: int,
    b_stride: int,
) -> None:
    if g_out.dim() != 2 or beta_out.shape != g_out.shape:
        raise RuntimeError("g_out/beta_out must have shape (rows, heads)")
    heads = g_out.shape[1]
    if neg_exp_a_log.shape != (heads,) or dt_bias.shape != (heads,):
        raise RuntimeError("per-head parameter shape mismatch")
    return None


def bf16_matvec(x: torch.Tensor, w: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute `out = x @ w.T` for BF16 `x (K,)` and `w (N, K)`."""
    if out is None:
        out = torch.empty((w.shape[0],), device=x.device, dtype=torch.bfloat16)
    ops.bf16_matvec(x, w, out)
    return out


def bf16_smallm_matmul(x: torch.Tensor, w: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute `out = x @ w.T` for BF16 `x (M,K)` and small `M`."""
    if out is None:
        out = torch.empty((x.shape[0], w.shape[0]), device=x.device, dtype=torch.bfloat16)
    ops.bf16_smallm_matmul(x, w, out)
    return out


def split_qkv_broadcast_bf16(
    packed: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    v_heads: int,
    head_dim: int,
    *,
    q: Optional[torch.Tensor] = None,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split packed Q/K/V and broadcast Q/K groups to `v_heads`."""
    rows = packed.shape[0]
    shape = (rows, v_heads, head_dim)
    if q is None:
        q = torch.empty(shape, device=packed.device, dtype=torch.bfloat16)
    if k is None:
        k = torch.empty(shape, device=packed.device, dtype=torch.bfloat16)
    if v is None:
        v = torch.empty(shape, device=packed.device, dtype=torch.bfloat16)
    ops.split_qkv_broadcast_bf16(
        packed, q, k, v, int(q_heads), int(kv_heads), int(v_heads), int(head_dim)
    )
    return q, k, v


def partial_rope_qk_bf16(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_dim: int,
    *,
    q_out: Optional[torch.Tensor] = None,
    k_out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply split-half RoPE to the first `rope_dim` channels of Q and K."""
    if q_out is None:
        q_out = torch.empty_like(q_in)
    if k_out is None:
        k_out = torch.empty_like(k_in)
    ops.partial_rope_qk_bf16(q_in, k_in, cos, sin, q_out, k_out, int(rope_dim))
    return q_out, k_out


def gated_delta_prepare_bf16(
    a: torch.Tensor,
    b: torch.Tensor,
    neg_exp_a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    heads: Optional[int] = None,
    a_stride: Optional[int] = None,
    b_stride: Optional[int] = None,
    g_out: Optional[torch.Tensor] = None,
    beta_out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare BF16 Gated DeltaNet `g` and `beta` tensors from projected a/b."""
    if heads is None:
        heads = neg_exp_a_log.shape[0]
    if a_stride is None:
        a_stride = a.shape[1]
    if b_stride is None:
        b_stride = b.shape[1]
    rows = a.shape[0]
    if g_out is None:
        g_out = torch.empty((rows, heads), device=a.device, dtype=torch.bfloat16)
    if beta_out is None:
        beta_out = torch.empty_like(g_out)
    ops.gated_delta_prepare_bf16(
        a, b, neg_exp_a_log, dt_bias, g_out, beta_out, int(a_stride), int(b_stride)
    )
    return g_out, beta_out


__all__ = [
    "bf16_matvec",
    "bf16_smallm_matmul",
    "gated_delta_prepare_bf16",
    "partial_rope_qk_bf16",
    "split_qkv_broadcast_bf16",
]
