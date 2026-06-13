# SPDX-License-Identifier: Apache-2.0
"""Optional native CUDA helpers for the FlashRT SM121 MSA package."""

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


__all__ = ["has_native_ops", "native_topk_from_scores"]
