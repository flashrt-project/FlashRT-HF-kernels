# SPDX-License-Identifier: Apache-2.0
"""MiniMaxAI/msa-compatible public interface for Blackwell decode sparse path."""

from __future__ import annotations

from typing import Optional

import torch

from .decode.topk_sparse import flash_decode_with_gqa_share_sparse


def _unsupported(name: str, reason: str):
    raise NotImplementedError(
        f"{name} is not implemented in flashrt/MiniMaxAI-msa-blackwell yet: {reason}"
    )


def _page_table_to_req_to_token(
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    max_seqlen_k: int,
    blk_kv: int,
) -> torch.Tensor:
    batch = int(page_table.shape[0])
    max_seqlen_k = int(max_seqlen_k)
    blk_kv = int(blk_kv)
    logical_pos = torch.arange(max_seqlen_k, device=page_table.device, dtype=torch.int64)
    page = logical_pos // blk_kv
    offset = logical_pos % blk_kv
    page = page.clamp(max=int(page_table.shape[1]) - 1)
    physical_page = page_table[:, page].to(torch.int64)
    physical_token = physical_page * blk_kv + offset[None, :]
    valid = logical_pos[None, :] < seqused_k.to(torch.int64)[:, None]
    return torch.where(valid, physical_token, torch.zeros_like(physical_token)).to(torch.int32).contiguous()


def _dense_q2k_indices(
    *,
    head_kv: int,
    batch: int,
    topk: int,
    seq_lens: torch.Tensor,
    blk_kv: int,
) -> torch.Tensor:
    idx = torch.full((head_kv, batch, topk), -1, dtype=torch.int32, device=seq_lens.device)
    blocks = torch.arange(topk, dtype=torch.int32, device=seq_lens.device)
    valid_blocks = (seq_lens.to(torch.int64) + int(blk_kv) - 1) // int(blk_kv)
    for b in range(batch):
        valid = blocks < valid_blocks[b]
        idx[:, b, valid] = blocks[valid]
    return idx


def _normalize_decode_q2k(
    q2k_indices: Optional[torch.Tensor],
    *,
    batch: int,
    seqlen_q: int,
    head_kv: int,
    seqused_k: torch.Tensor,
    blk_kv: int,
) -> torch.Tensor:
    if q2k_indices is None:
        topk = int((int(seqused_k.max().item()) + int(blk_kv) - 1) // int(blk_kv))
        return _dense_q2k_indices(
            head_kv=head_kv,
            batch=batch,
            topk=topk,
            seq_lens=seqused_k,
            blk_kv=blk_kv,
        )
    if q2k_indices.dtype != torch.int32:
        raise TypeError("q2k_indices must be torch.int32")
    if q2k_indices.ndim != 3:
        raise ValueError("q2k_indices must have shape [Hkv, total_q, topK]")
    if q2k_indices.shape[0] != head_kv:
        raise ValueError("q2k_indices first dimension must match Hkv")
    if int(q2k_indices.shape[1]) != batch * seqlen_q:
        raise ValueError("q2k_indices second dimension must equal batch * seqlen_q")
    if seqlen_q != 1:
        raise NotImplementedError("Blackwell decode wrapper currently supports seqlen_q=1")
    return q2k_indices[:, ::seqlen_q, :].contiguous()


def sparse_decode_atten_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: Optional[torch.Tensor] = None,
    *,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    seqlen_q: int,
    max_seqlen_k: int,
    blk_kv: int = 128,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
    return_softmax_lse: bool = False,
    schedule: Optional[object] = None,
    O_partial: Optional[torch.Tensor] = None,
    LSE_partial: Optional[torch.Tensor] = None,
):
    """Run MiniMax paged sparse decode on Blackwell.

    This is compatible with the upstream MiniMaxAI/msa decode-facing API name,
    but accepts the Blackwell package's validated BF16/FP16 decode path.  It
    supports single-token decode (`seqlen_q=1`) with paged K/V
    `[num_pages, Hkv, blk_kv, 128]` and optional sparse `q2k_indices`.
    """

    if schedule is not None or O_partial is not None or LSE_partial is not None:
        raise NotImplementedError("explicit SM100 decode schedules/partials are not used by the Blackwell wrapper")
    if return_softmax_lse:
        raise NotImplementedError("return_softmax_lse is not implemented for the Blackwell wrapper")
    if not causal:
        raise NotImplementedError("Blackwell decode wrapper currently supports causal=True")
    if int(seqlen_q) != 1:
        raise NotImplementedError("Blackwell decode wrapper currently supports seqlen_q=1")
    if q.ndim != 3:
        raise ValueError("q must have shape [batch * seqlen_q, Hq, D]")
    if k.ndim != 4 or v.ndim != 4:
        raise ValueError("k and v must have shape [num_pages, Hkv, blk_kv, D]")
    if k.shape != v.shape:
        raise ValueError("k and v must have the same shape")
    if q.device != k.device or q.device != v.device or q.device != page_table.device:
        raise ValueError("q, k, v, and page_table must be on the same device")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("Blackwell decode wrapper supports BF16/FP16 q")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("q, k, and v must have the same dtype for the Blackwell decode wrapper")
    if q.shape[-1] != 128 or k.shape[-1] != 128:
        raise NotImplementedError("Blackwell decode wrapper currently supports head_dim=128")
    if int(k.shape[2]) != int(blk_kv):
        raise ValueError("k.shape[2] must equal blk_kv")
    if page_table.dtype != torch.int32 or seqused_k.dtype != torch.int32:
        raise TypeError("page_table and seqused_k must be torch.int32")
    if page_table.ndim != 2:
        raise ValueError("page_table must have shape [batch, max_num_pages_per_seq]")
    batch = int(page_table.shape[0])
    if int(q.shape[0]) != batch:
        raise ValueError("for seqlen_q=1, q.shape[0] must equal batch")

    head_kv = int(k.shape[1])
    q2k = _normalize_decode_q2k(
        q2k_indices,
        batch=batch,
        seqlen_q=int(seqlen_q),
        head_kv=head_kv,
        seqused_k=seqused_k,
        blk_kv=int(blk_kv),
    )
    req_to_token = _page_table_to_req_to_token(page_table, seqused_k, int(max_seqlen_k), int(blk_kv))
    slot_ids = torch.arange(batch, device=q.device, dtype=torch.int64)
    k_cache = k.transpose(1, 2).reshape(k.shape[0] * k.shape[2], k.shape[1], k.shape[3]).contiguous()
    v_cache = v.transpose(1, 2).reshape(v.shape[0] * v.shape[2], v.shape[1], v.shape[3]).contiguous()
    return flash_decode_with_gqa_share_sparse(
        q.contiguous(),
        None,
        k_cache,
        v_cache,
        req_to_token,
        seqused_k.contiguous(),
        slot_ids,
        int(blk_kv),
        q2k,
        sm_scale=softmax_scale,
    )


class SparseDecodePagedAttentionWrapper:
    def __init__(self, *, blk_kv: int = 128, causal: bool = True):
        self.blk_kv = int(blk_kv)
        self.causal = bool(causal)
        self.page_table: Optional[torch.Tensor] = None
        self.seqused_k: Optional[torch.Tensor] = None
        self.seqlen_q: Optional[int] = None
        self.max_seqlen_k: Optional[int] = None
        self.q2k_indices: Optional[torch.Tensor] = None

    def plan(
        self,
        *,
        page_table: torch.Tensor,
        seqused_k: torch.Tensor,
        seqlen_q: int,
        max_seqlen_k: int,
        q2k_indices: Optional[torch.Tensor] = None,
        num_qo_heads: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = 128,
        enable_cuda_graph: bool = False,
        max_grid_size: Optional[int] = None,
        fixed_split_size: Optional[int] = None,
        disable_split_kv: bool = False,
    ) -> "SparseDecodePagedAttentionWrapper":
        if head_dim is not None and int(head_dim) != 128:
            raise NotImplementedError("Blackwell decode wrapper currently supports head_dim=128")
        self.page_table = page_table.contiguous()
        self.seqused_k = seqused_k.contiguous()
        self.seqlen_q = int(seqlen_q)
        self.max_seqlen_k = int(max_seqlen_k)
        self.q2k_indices = None if q2k_indices is None else q2k_indices.contiguous()
        return self

    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        softmax_scale: Optional[float] = None,
        return_softmax_lse: bool = False,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
    ):
        if self.page_table is None or self.seqused_k is None:
            raise RuntimeError("decode wrapper must be planned before run")
        result = sparse_decode_atten_func(
            q,
            k,
            v,
            self.q2k_indices,
            page_table=self.page_table,
            seqused_k=self.seqused_k,
            seqlen_q=self.seqlen_q,
            max_seqlen_k=self.max_seqlen_k,
            blk_kv=self.blk_kv,
            causal=self.causal,
            softmax_scale=softmax_scale,
            return_softmax_lse=return_softmax_lse,
        )
        if out is not None:
            out.copy_(result)
            return out
        return result


def sparse_atten_func(*args, **kwargs):
    _unsupported("sparse_atten_func", "SM100 CSR prefill attention is not ported to Blackwell in this package")


def sparse_atten_nvfp4_kv_func(*args, **kwargs):
    _unsupported("sparse_atten_nvfp4_kv_func", "SM100 NVFP4 CSR prefill attention is not ported to Blackwell in this package")


def fp4_indexer_block_scores(*args, **kwargs):
    _unsupported("fp4_indexer_block_scores", "SM100 FP4 CUTE indexer is not ported to Blackwell in this package")


__all__ = [
    "sparse_atten_func",
    "sparse_atten_nvfp4_kv_func",
    "sparse_decode_atten_func",
    "SparseDecodePagedAttentionWrapper",
    "fp4_indexer_block_scores",
]
