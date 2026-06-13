# SPDX-License-Identifier: Apache-2.0
"""Optional native CUDA helpers for the FlashRT Blackwell MSA package."""

from __future__ import annotations

import torch

try:
    from ._ops import add_op_namespace_prefix, ops

    _HAS_NATIVE_OPS = True
except Exception:
    add_op_namespace_prefix = None
    ops = None
    _HAS_NATIVE_OPS = False


def has_native_ops() -> bool:
    return _HAS_NATIVE_OPS


if _HAS_NATIVE_OPS:

    @torch.library.register_fake(add_op_namespace_prefix("msa_topk_from_scores"))
    def _msa_topk_from_scores_fake(
        score: torch.Tensor,
        seq_lens: torch.Tensor,
        block_size: int,
        topk: int,
        topk_idx: torch.Tensor,
    ) -> None:
        if score.dim() != 3:
            raise RuntimeError("score must have shape (heads, batch, max_blocks)")
        heads, batch, _ = score.shape
        if seq_lens.shape != (batch,):
            raise RuntimeError("seq_lens must have shape (batch,)")
        if topk_idx.shape != (heads, batch, topk):
            raise RuntimeError("topk_idx must have shape (heads, batch, topk)")
        return None


def native_topk_from_scores(
    score: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    topk: int,
) -> torch.Tensor:
    """Return top-k sparse block ids from a score tensor."""

    if not _HAS_NATIVE_OPS:
        raise RuntimeError("native MiniMax MSA ops are not available in source-tree mode")
    if not score.is_contiguous():
        score = score.contiguous()
    if not seq_lens.is_contiguous():
        seq_lens = seq_lens.contiguous()
    out = torch.empty(
        (score.shape[0], score.shape[1], int(topk)),
        device=score.device,
        dtype=torch.int32,
    )
    ops.msa_topk_from_scores(score, seq_lens, int(block_size), int(topk), out)
    return out


if _HAS_NATIVE_OPS:

    @torch.library.register_fake(
        add_op_namespace_prefix("msa_decode_sparse_attn_mma_paged")
    )
    def _msa_decode_sparse_attn_mma_paged_fake(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        seq_lens: torch.Tensor,
        slot_ids: torch.Tensor,
        topk_idx: torch.Tensor,
        block_size: int,
        sm_scale: float,
        out: torch.Tensor,
    ) -> None:
        return None


def native_decode_mma_supported(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    *,
    block_size: int,
) -> bool:
    """Whether the tensor-core decode kernel applies to these inputs.

    The mma kernel is specialized for the M3 MSA shape: head_dim 128, GQA
    group (Hq/Hkv) 16, bf16, and a block size that is a multiple of 64.
    """

    if not _HAS_NATIVE_OPS:
        return False
    if q.dtype is not torch.bfloat16 or k_cache.dtype is not torch.bfloat16:
        return False
    if q.dim() != 3 or k_cache.dim() != 3:
        return False
    head_dim = int(q.shape[-1])
    num_q_heads = int(q.shape[1])
    num_kv_heads = int(k_cache.shape[1])
    if head_dim != 128 or num_kv_heads <= 0:
        return False
    if num_q_heads % num_kv_heads != 0 or num_q_heads // num_kv_heads != 16:
        return False
    if int(block_size) % 64 != 0:
        return False
    return True


def native_decode_sparse_attn_mma_paged(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,
    slot_ids: torch.Tensor,
    block_size: int,
    topk_idx: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Paged tensor-core block-sparse GQA decode. Returns [B, Hq, D] bf16."""

    if not _HAS_NATIVE_OPS:
        raise RuntimeError("native MiniMax MSA ops are not available in source-tree mode")
    out = torch.empty(
        (int(q.shape[0]), int(q.shape[1]), int(q.shape[2])),
        device=q.device,
        dtype=q.dtype,
    )
    ops.msa_decode_sparse_attn_mma_paged(
        q.contiguous(),
        k_cache.contiguous(),
        v_cache.contiguous(),
        req_to_token.contiguous(),
        seq_lens.contiguous(),
        slot_ids.contiguous(),
        topk_idx.contiguous(),
        int(block_size),
        float(sm_scale),
        out,
    )
    return out


if _HAS_NATIVE_OPS:

    @torch.library.register_fake(
        add_op_namespace_prefix("msa_indexer_block_scores")
    )
    def _msa_indexer_block_scores_fake(
        q: torch.Tensor,
        k_pages: torch.Tensor,
        batch_of_q: torch.Tensor,
        cu_q: torch.Tensor,
        cu_k: torch.Tensor,
        cu_pages: torch.Tensor,
        kv_indices: torch.Tensor,
        causal: int,
        scores: torch.Tensor,
    ) -> None:
        return None


def native_indexer_block_scores(
    q: torch.Tensor,
    k_pages: torch.Tensor,
    batch_of_q: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    cu_pages: torch.Tensor,
    kv_indices: torch.Tensor,
    *,
    max_blocks: int,
    causal: bool,
) -> torch.Tensor:
    """Block-max QK indexer scores. q/k_pages are bf16 (already dequantized).

    Returns [Hq, max_blocks, total_q] f32 (-inf where no key is visible).
    """

    if not _HAS_NATIVE_OPS:
        raise RuntimeError("native MiniMax MSA ops are not available in source-tree mode")
    total_q = int(q.shape[0])
    hq = int(q.shape[1])
    scores = torch.full(
        (hq, int(max_blocks), total_q),
        float("-inf"),
        dtype=torch.float32,
        device=q.device,
    )
    ops.msa_indexer_block_scores(
        q.contiguous(),
        k_pages.contiguous(),
        batch_of_q.contiguous(),
        cu_q.contiguous(),
        cu_k.contiguous(),
        cu_pages.contiguous(),
        kv_indices.contiguous(),
        1 if causal else 0,
        scores,
    )
    return scores


if _HAS_NATIVE_OPS:

    @torch.library.register_fake(
        add_op_namespace_prefix("msa_nvfp4_dequant_swizzled_to_bf16")
    )
    def _msa_nvfp4_dequant_swizzled_to_bf16_fake(
        packed: torch.Tensor,
        scale_128x4: torch.Tensor,
        global_scale: float,
        out: torch.Tensor,
    ) -> None:
        return None


def native_nvfp4_dequant_swizzled_to_bf16(
    packed: torch.Tensor,
    scale_128x4: torch.Tensor,
    global_scale: torch.Tensor | float,
) -> torch.Tensor:
    """Expand swizzled NVFP4 weights/KV to dense BF16 with the native CUDA path."""

    if not _HAS_NATIVE_OPS:
        raise RuntimeError("native MiniMax MSA ops are not available in source-tree mode")
    if packed.dtype is not torch.uint8:
        raise TypeError("packed must be torch.uint8")
    if scale_128x4.dtype is not torch.uint8:
        raise TypeError("scale_128x4 must be torch.uint8")
    original_shape = tuple(int(v) for v in packed.shape[:-1]) + (int(packed.shape[-1]) * 2,)
    out = torch.empty(original_shape, device=packed.device, dtype=torch.bfloat16)
    if isinstance(global_scale, torch.Tensor):
        global_scale_value = float(global_scale.reshape(-1)[0].item())
    else:
        global_scale_value = float(global_scale)
    ops.msa_nvfp4_dequant_swizzled_to_bf16(
        packed.contiguous(),
        scale_128x4.contiguous(),
        global_scale_value,
        out,
    )
    return out


__all__ = [
    "has_native_ops",
    "native_topk_from_scores",
    "native_decode_mma_supported",
    "native_decode_sparse_attn_mma_paged",
    "native_indexer_block_scores",
    "native_nvfp4_dequant_swizzled_to_bf16",
]
