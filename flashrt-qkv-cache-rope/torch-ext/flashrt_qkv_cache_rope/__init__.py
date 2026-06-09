"""FlashRT QKV split, Q/K RMSNorm, and RoPE kernels."""

from __future__ import annotations

import torch

from ._ops import add_op_namespace_prefix, ops


def _check_decode_rope(x: torch.Tensor, weight: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, out: torch.Tensor, name: str) -> None:
    if x.dim() != 2 or x.shape[1] != 128:
        raise RuntimeError(f"{name} must have shape (heads, 128)")
    if weight.shape != (128,):
        raise RuntimeError("norm weight must have shape (128,)")
    if cos.shape != (64,) or sin.shape != (64,):
        raise RuntimeError("cos and sin must have shape (64,)")
    if out.shape != x.shape:
        raise RuntimeError("out must have the same shape as input")


@torch.library.register_fake(add_op_namespace_prefix("decode_q_norm_rope_stage_bf16"))
def _decode_q_norm_rope_stage_bf16_fake(
    q_pre: torch.Tensor,
    q_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    q_out: torch.Tensor,
) -> None:
    _check_decode_rope(q_pre, q_norm_weight, cos, sin, q_out, "q_pre")
    return None


@torch.library.register_fake(add_op_namespace_prefix("decode_k_norm_rope_kvwrite_bf16"))
def _decode_k_norm_rope_kvwrite_bf16_fake(
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


@torch.library.register_fake(add_op_namespace_prefix("decode_k_norm_rope_kvwrite_devpos_bf16"))
def _decode_k_norm_rope_kvwrite_devpos_bf16_fake(
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


@torch.library.register_fake(add_op_namespace_prefix("qkv_split_rope_kvcache_bf16"))
def _qkv_split_rope_kvcache_bf16_fake(
    packed_qkv: torch.Tensor,
    rope: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    cache_offset: int,
    q_out: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> None:
    _check_packed_gqa_qkv(packed_qkv, q_heads, kv_heads, head_dim)
    batch, seq_len, _ = packed_qkv.shape
    if rope.dim() != 2 or rope.shape[0] < seq_len or rope.shape[1] != head_dim:
        raise RuntimeError("rope must have shape (>= seq_len, head_dim)")
    if q_out.shape != (batch, seq_len, q_heads, head_dim):
        raise RuntimeError("q_out must have shape (batch, seq_len, q_heads, head_dim)")
    if k_cache.dim() != 4 or k_cache.shape[0] != batch or k_cache.shape[2:] != (kv_heads, head_dim):
        raise RuntimeError("k_cache must have shape (batch, max_seq_len, kv_heads, head_dim)")
    if v_cache.shape != k_cache.shape:
        raise RuntimeError("v_cache must have the same shape as k_cache")
    if cache_offset < 0 or cache_offset + seq_len > k_cache.shape[1]:
        raise RuntimeError("cache_offset + seq_len must be within k_cache.shape[1]")
    return None


@torch.library.register_fake(add_op_namespace_prefix("qkv_split_bf16"))
def _qkv_split_bf16_fake(
    packed_qkv: torch.Tensor,
    heads: int,
    head_dim: int,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
) -> None:
    if packed_qkv.dim() != 3:
        raise RuntimeError("packed_qkv must have shape (batch, seq_len, 3 * heads * head_dim)")
    batch, seq_len, cols = packed_qkv.shape
    if cols != 3 * heads * head_dim:
        raise RuntimeError("packed_qkv.shape[2] must be 3 * heads * head_dim")
    out_shape = (batch, seq_len, heads, head_dim)
    if q_out.shape != out_shape or k_out.shape != out_shape or v_out.shape != out_shape:
        raise RuntimeError("q_out, k_out, and v_out must have shape (batch, seq_len, heads, head_dim)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("qkv_split_norm_rope_bf16"))
def _qkv_split_norm_rope_bf16_fake(
    packed_qkv: torch.Tensor,
    norm_q_weight: torch.Tensor,
    norm_k_weight: torch.Tensor,
    freqs_re: torch.Tensor,
    freqs_im: torch.Tensor,
    heads: int,
    head_dim: int,
    rope_seq_len: int,
    eps: float,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
) -> None:
    if packed_qkv.dim() != 3:
        raise RuntimeError("packed_qkv must have shape (batch, seq_len, 3 * heads * head_dim)")
    batch, seq_len, cols = packed_qkv.shape
    dim = heads * head_dim
    if cols != 3 * dim:
        raise RuntimeError("packed_qkv.shape[2] must be 3 * heads * head_dim")
    if norm_q_weight.shape != (dim,) or norm_k_weight.shape != (dim,):
        raise RuntimeError("norm weights must have shape (heads * head_dim,)")
    if freqs_re.dim() != 2 or freqs_re.shape[1] != head_dim // 2:
        raise RuntimeError("freqs_re must have shape (rope_seq_len, head_dim / 2)")
    if freqs_im.shape != freqs_re.shape:
        raise RuntimeError("freqs_im must have the same shape as freqs_re")
    if q_out.shape != (batch, seq_len, heads, head_dim):
        raise RuntimeError("q_out must have shape (batch, seq_len, heads, head_dim)")
    if k_out.shape != q_out.shape:
        raise RuntimeError("k_out must have the same shape as q_out")
    return None


@torch.library.register_fake(add_op_namespace_prefix("qkv_split_bias_norm_rope_v_bf16"))
def _qkv_split_bias_norm_rope_v_bf16_fake(
    packed_qkv: torch.Tensor,
    qkv_bias: torch.Tensor,
    norm_q_weight: torch.Tensor,
    norm_k_weight: torch.Tensor,
    freqs_re: torch.Tensor,
    freqs_im: torch.Tensor,
    heads: int,
    head_dim: int,
    rope_seq_len: int,
    eps: float,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
) -> None:
    _check_packed_qkv(packed_qkv, norm_q_weight, norm_k_weight, heads, head_dim)
    batch, seq_len, _ = packed_qkv.shape
    dim = heads * head_dim
    if qkv_bias.shape != (3 * dim,):
        raise RuntimeError("qkv_bias must have shape (3 * heads * head_dim,)")
    _check_freqs(freqs_re, freqs_im, head_dim, rope_seq_len)
    out_shape = (batch, seq_len, heads, head_dim)
    if q_out.shape != out_shape or k_out.shape != out_shape or v_out.shape != out_shape:
        raise RuntimeError("q_out, k_out, and v_out must have shape (batch, seq_len, heads, head_dim)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("qkv_split_bias_norm_rope_v_cat_bf16"))
def _qkv_split_bias_norm_rope_v_cat_bf16_fake(
    packed_qkv: torch.Tensor,
    qkv_bias: torch.Tensor,
    norm_q_weight: torch.Tensor,
    norm_k_weight: torch.Tensor,
    freqs_re: torch.Tensor,
    freqs_im: torch.Tensor,
    heads: int,
    head_dim: int,
    video_offset: int,
    rope_seq_len: int,
    eps: float,
    q_cat_out: torch.Tensor,
    k_cat_out: torch.Tensor,
    v_cat_out: torch.Tensor,
) -> None:
    _check_packed_qkv(packed_qkv, norm_q_weight, norm_k_weight, heads, head_dim)
    batch, seq_len, _ = packed_qkv.shape
    dim = heads * head_dim
    if qkv_bias.shape != (3 * dim,):
        raise RuntimeError("qkv_bias must have shape (3 * heads * head_dim,)")
    if q_cat_out.dim() != 4:
        raise RuntimeError("q_cat_out must have shape (batch, total_seq_len, heads, head_dim)")
    total_seq_len = q_cat_out.shape[1]
    if video_offset < 0 or video_offset + seq_len > total_seq_len:
        raise RuntimeError("video_offset + packed_qkv.shape[1] must be within q_cat_out.shape[1]")
    _check_freqs(freqs_re, freqs_im, head_dim, rope_seq_len)
    out_shape = (batch, total_seq_len, heads, head_dim)
    if q_cat_out.shape != out_shape or k_cat_out.shape != out_shape or v_cat_out.shape != out_shape:
        raise RuntimeError("cat outputs must have shape (batch, total_seq_len, heads, head_dim)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("qkv_split_joint3_cat_bf16"))
def _qkv_split_joint3_cat_bf16_fake(
    packed_v: torch.Tensor,
    qkv_v_bias: torch.Tensor,
    norm_v_q_weight: torch.Tensor,
    norm_v_k_weight: torch.Tensor,
    freqs_re: torch.Tensor,
    freqs_im: torch.Tensor,
    packed_a: torch.Tensor,
    norm_a_q_weight: torch.Tensor,
    norm_a_k_weight: torch.Tensor,
    packed_u: torch.Tensor,
    norm_u_q_weight: torch.Tensor,
    norm_u_k_weight: torch.Tensor,
    heads: int,
    head_dim: int,
    rope_seq_len: int,
    eps_v: float,
    eps_a: float,
    eps_u: float,
    q_cat_out: torch.Tensor,
    k_cat_out: torch.Tensor,
    v_cat_out: torch.Tensor,
) -> None:
    _check_packed_qkv(packed_v, norm_v_q_weight, norm_v_k_weight, heads, head_dim)
    _check_packed_qkv(packed_a, norm_a_q_weight, norm_a_k_weight, heads, head_dim)
    _check_packed_qkv(packed_u, norm_u_q_weight, norm_u_k_weight, heads, head_dim)
    batch = packed_v.shape[0]
    if batch != 1 or packed_a.shape[0] != batch or packed_u.shape[0] != batch:
        raise RuntimeError("qkv_split_joint3_cat_bf16 currently supports batch == 1")
    total_seq_len = packed_v.shape[1] + packed_a.shape[1] + packed_u.shape[1]
    dim = heads * head_dim
    if qkv_v_bias.shape != (3 * dim,):
        raise RuntimeError("qkv_v_bias must have shape (3 * heads * head_dim,)")
    _check_freqs(freqs_re, freqs_im, head_dim, rope_seq_len)
    out_shape = (batch, total_seq_len, heads, head_dim)
    if q_cat_out.shape != out_shape or k_cat_out.shape != out_shape or v_cat_out.shape != out_shape:
        raise RuntimeError("cat outputs must have shape (1, L_v + L_a + L_u, heads, head_dim)")
    return None


def _check_packed_qkv(
    packed_qkv: torch.Tensor,
    norm_q_weight: torch.Tensor,
    norm_k_weight: torch.Tensor,
    heads: int,
    head_dim: int,
) -> None:
    if packed_qkv.dim() != 3:
        raise RuntimeError("packed_qkv must have shape (batch, seq_len, 3 * heads * head_dim)")
    dim = heads * head_dim
    if head_dim % 2 != 0 or packed_qkv.shape[2] != 3 * dim:
        raise RuntimeError("packed_qkv.shape[2] must be 3 * heads * head_dim and head_dim must be even")
    if norm_q_weight.shape != (dim,) or norm_k_weight.shape != (dim,):
        raise RuntimeError("norm weights must have shape (heads * head_dim,)")


def _check_packed_gqa_qkv(
    packed_qkv: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> None:
    if packed_qkv.dim() != 3:
        raise RuntimeError("packed_qkv must have shape (batch, seq_len, (q_heads + 2 * kv_heads) * head_dim)")
    if q_heads <= 0 or kv_heads <= 0 or head_dim <= 0 or head_dim % 2 != 0:
        raise RuntimeError("q_heads, kv_heads, and even head_dim must be positive")
    expected = (q_heads + 2 * kv_heads) * head_dim
    if packed_qkv.shape[2] != expected:
        raise RuntimeError("packed_qkv.shape[2] must be (q_heads + 2 * kv_heads) * head_dim")


def _check_freqs(
    freqs_re: torch.Tensor,
    freqs_im: torch.Tensor,
    head_dim: int,
    rope_seq_len: int,
) -> None:
    if freqs_re.dim() != 2 or freqs_re.shape[1] != head_dim // 2:
        raise RuntimeError("freqs_re must have shape (rope_seq_len, head_dim / 2)")
    if freqs_im.shape != freqs_re.shape:
        raise RuntimeError("freqs_im must have the same shape as freqs_re")
    if rope_seq_len < 0 or freqs_re.shape[0] < rope_seq_len:
        raise RuntimeError("freqs_re must have at least rope_seq_len rows")


def qkv_split_norm_rope_bf16(
    packed_qkv: torch.Tensor,
    norm_q_weight: torch.Tensor,
    norm_k_weight: torch.Tensor,
    freqs_re: torch.Tensor,
    freqs_im: torch.Tensor,
    heads: int,
    head_dim: int,
    rope_seq_len: int | None = None,
    eps: float = 1e-6,
    q_out: torch.Tensor | None = None,
    k_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split packed QKV, RMSNorm Q/K, apply RoPE, and return Q/K tensors.

    ``packed_qkv`` has shape ``(batch, seq_len, 3 * heads * head_dim)``.
    Outputs have shape ``(batch, seq_len, heads, head_dim)``.
    """

    if rope_seq_len is None:
        rope_seq_len = packed_qkv.shape[1]
    out_shape = (packed_qkv.shape[0], packed_qkv.shape[1], heads, head_dim)
    if q_out is None:
        q_out = torch.empty(out_shape, device=packed_qkv.device, dtype=torch.bfloat16)
    if k_out is None:
        k_out = torch.empty(out_shape, device=packed_qkv.device, dtype=torch.bfloat16)
    ops.qkv_split_norm_rope_bf16(
        packed_qkv,
        norm_q_weight,
        norm_k_weight,
        freqs_re,
        freqs_im,
        int(heads),
        int(head_dim),
        int(rope_seq_len),
        float(eps),
        q_out,
        k_out,
    )
    return q_out, k_out


def qkv_split_rope_kvcache_bf16(
    packed_qkv: torch.Tensor,
    rope: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    cache_offset: int,
    q_out: torch.Tensor | None = None,
    k_cache: torch.Tensor | None = None,
    v_cache: torch.Tensor | None = None,
    max_seq_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split GQA packed QKV, apply interleaved RoPE, and write K/V cache.

    ``packed_qkv`` has shape ``(batch, seq_len, (q_heads + 2 * kv_heads) * head_dim)``.
    ``rope`` has BF16 interleaved ``[cos0, sin0, cos1, sin1, ...]`` rows with
    shape ``(>= seq_len, head_dim)``. ``q_out`` has shape
    ``(batch, seq_len, q_heads, head_dim)``. K/V are written in-place into
    ``(batch, max_seq_len, kv_heads, head_dim)`` caches starting at
    ``cache_offset``.
    """

    batch, seq_len, _ = packed_qkv.shape
    if q_out is None:
        q_out = torch.empty(
            (batch, seq_len, q_heads, head_dim),
            device=packed_qkv.device,
            dtype=torch.bfloat16,
        )
    if k_cache is None or v_cache is None:
        if max_seq_len is None:
            max_seq_len = cache_offset + seq_len
        cache_shape = (batch, int(max_seq_len), kv_heads, head_dim)
        if k_cache is None:
            k_cache = torch.empty(cache_shape, device=packed_qkv.device, dtype=torch.bfloat16)
        if v_cache is None:
            v_cache = torch.empty(cache_shape, device=packed_qkv.device, dtype=torch.bfloat16)
    ops.qkv_split_rope_kvcache_bf16(
        packed_qkv,
        rope,
        int(q_heads),
        int(kv_heads),
        int(head_dim),
        int(cache_offset),
        q_out,
        k_cache,
        v_cache,
    )
    return q_out, k_cache, v_cache


def qkv_split_bf16(
    packed_qkv: torch.Tensor,
    heads: int,
    head_dim: int,
    q_out: torch.Tensor | None = None,
    k_out: torch.Tensor | None = None,
    v_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split packed BF16 QKV into Q/K/V tensors.

    ``packed_qkv`` has shape ``(batch, seq_len, 3 * heads * head_dim)``.
    Outputs have shape ``(batch, seq_len, heads, head_dim)``.
    """

    out_shape = (packed_qkv.shape[0], packed_qkv.shape[1], heads, head_dim)
    if q_out is None:
        q_out = torch.empty(out_shape, device=packed_qkv.device, dtype=torch.bfloat16)
    if k_out is None:
        k_out = torch.empty(out_shape, device=packed_qkv.device, dtype=torch.bfloat16)
    if v_out is None:
        v_out = torch.empty(out_shape, device=packed_qkv.device, dtype=torch.bfloat16)
    ops.qkv_split_bf16(packed_qkv, int(heads), int(head_dim), q_out, k_out, v_out)
    return q_out, k_out, v_out


def decode_q_norm_rope_stage_bf16(
    q_pre: torch.Tensor,
    q_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-6,
    q_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """RMSNorm Q, apply rotate-half RoPE, and write a decode Q staging buffer.

    The decode path is fixed to ``head_dim == 128``. ``cos`` and ``sin`` have
    shape ``(64,)`` and dtype BF16.
    """

    if q_out is None:
        q_out = torch.empty_like(q_pre)
    ops.decode_q_norm_rope_stage_bf16(
        q_pre, q_norm_weight, cos, sin, float(eps), q_out
    )
    return q_out


def decode_k_norm_rope_kvwrite_bf16(
    k_pre: torch.Tensor,
    v_pre: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-6,
    k_cache_dst: torch.Tensor | None = None,
    v_cache_dst: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm K, apply rotate-half RoPE, and write one KV cache slot."""

    if k_cache_dst is None:
        k_cache_dst = torch.empty_like(k_pre)
    if v_cache_dst is None:
        v_cache_dst = torch.empty_like(v_pre)
    ops.decode_k_norm_rope_kvwrite_bf16(
        k_pre, v_pre, k_norm_weight, cos, sin, float(eps), k_cache_dst, v_cache_dst
    )
    return k_cache_dst, v_cache_dst


def decode_k_norm_rope_kvwrite_devpos_bf16(
    k_pre: torch.Tensor,
    v_pre: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cur_pos: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write one KV cache slot selected by device int32 ``cur_pos``."""

    ops.decode_k_norm_rope_kvwrite_devpos_bf16(
        k_pre, v_pre, k_norm_weight, cos, sin, cur_pos, float(eps), k_cache, v_cache
    )
    return k_cache, v_cache


def qkv_split_bias_norm_rope_v_bf16(
    packed_qkv: torch.Tensor,
    qkv_bias: torch.Tensor,
    norm_q_weight: torch.Tensor,
    norm_k_weight: torch.Tensor,
    freqs_re: torch.Tensor,
    freqs_im: torch.Tensor,
    heads: int,
    head_dim: int,
    rope_seq_len: int | None = None,
    eps: float = 1e-6,
    q_out: torch.Tensor | None = None,
    k_out: torch.Tensor | None = None,
    v_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bias QKV, RMSNorm Q/K, apply RoPE, and materialize Q/K/V."""

    if rope_seq_len is None:
        rope_seq_len = packed_qkv.shape[1]
    out_shape = (packed_qkv.shape[0], packed_qkv.shape[1], heads, head_dim)
    if q_out is None:
        q_out = torch.empty(out_shape, device=packed_qkv.device, dtype=torch.bfloat16)
    if k_out is None:
        k_out = torch.empty(out_shape, device=packed_qkv.device, dtype=torch.bfloat16)
    if v_out is None:
        v_out = torch.empty(out_shape, device=packed_qkv.device, dtype=torch.bfloat16)
    ops.qkv_split_bias_norm_rope_v_bf16(
        packed_qkv,
        qkv_bias,
        norm_q_weight,
        norm_k_weight,
        freqs_re,
        freqs_im,
        int(heads),
        int(head_dim),
        int(rope_seq_len),
        float(eps),
        q_out,
        k_out,
        v_out,
    )
    return q_out, k_out, v_out


def qkv_split_bias_norm_rope_v_cat_bf16(
    packed_qkv: torch.Tensor,
    qkv_bias: torch.Tensor,
    norm_q_weight: torch.Tensor,
    norm_k_weight: torch.Tensor,
    freqs_re: torch.Tensor,
    freqs_im: torch.Tensor,
    heads: int,
    head_dim: int,
    video_offset: int,
    q_cat_out: torch.Tensor,
    k_cat_out: torch.Tensor,
    v_cat_out: torch.Tensor,
    rope_seq_len: int | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Write a biased video QKV segment directly into joint Q/K/V workspaces."""

    if rope_seq_len is None:
        rope_seq_len = packed_qkv.shape[1]
    ops.qkv_split_bias_norm_rope_v_cat_bf16(
        packed_qkv,
        qkv_bias,
        norm_q_weight,
        norm_k_weight,
        freqs_re,
        freqs_im,
        int(heads),
        int(head_dim),
        int(video_offset),
        int(rope_seq_len),
        float(eps),
        q_cat_out,
        k_cat_out,
        v_cat_out,
    )
    return q_cat_out, k_cat_out, v_cat_out


def qkv_split_joint3_cat_bf16(
    packed_v: torch.Tensor,
    qkv_v_bias: torch.Tensor,
    norm_v_q_weight: torch.Tensor,
    norm_v_k_weight: torch.Tensor,
    freqs_re: torch.Tensor,
    freqs_im: torch.Tensor,
    packed_a: torch.Tensor,
    norm_a_q_weight: torch.Tensor,
    norm_a_k_weight: torch.Tensor,
    packed_u: torch.Tensor,
    norm_u_q_weight: torch.Tensor,
    norm_u_k_weight: torch.Tensor,
    heads: int,
    head_dim: int,
    q_cat_out: torch.Tensor,
    k_cat_out: torch.Tensor,
    v_cat_out: torch.Tensor,
    rope_seq_len: int | None = None,
    eps_v: float = 1e-6,
    eps_a: float = 1e-6,
    eps_u: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse video/action/und QKV postprocess into one joint Q/K/V workspace."""

    if rope_seq_len is None:
        rope_seq_len = packed_v.shape[1]
    ops.qkv_split_joint3_cat_bf16(
        packed_v,
        qkv_v_bias,
        norm_v_q_weight,
        norm_v_k_weight,
        freqs_re,
        freqs_im,
        packed_a,
        norm_a_q_weight,
        norm_a_k_weight,
        packed_u,
        norm_u_q_weight,
        norm_u_k_weight,
        int(heads),
        int(head_dim),
        int(rope_seq_len),
        float(eps_v),
        float(eps_a),
        float(eps_u),
        q_cat_out,
        k_cat_out,
        v_cat_out,
    )
    return q_cat_out, k_cat_out, v_cat_out


__all__ = [
    "decode_q_norm_rope_stage_bf16",
    "decode_k_norm_rope_kvwrite_bf16",
    "decode_k_norm_rope_kvwrite_devpos_bf16",
    "qkv_split_bf16",
    "qkv_split_rope_kvcache_bf16",
    "qkv_split_norm_rope_bf16",
    "qkv_split_bias_norm_rope_v_bf16",
    "qkv_split_bias_norm_rope_v_cat_bf16",
    "qkv_split_joint3_cat_bf16",
]
