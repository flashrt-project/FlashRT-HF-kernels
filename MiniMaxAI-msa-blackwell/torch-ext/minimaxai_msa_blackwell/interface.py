# SPDX-License-Identifier: Apache-2.0
"""MiniMaxAI/msa-compatible public interface for Blackwell sparse MSA paths."""

from __future__ import annotations

from typing import Optional

import torch

from ._native import has_native_ops, native_indexer_block_scores
from .decode.topk_sparse import flash_decode_with_gqa_share_sparse
from .prefill.topk_sparse import flash_prefill_with_gqa_share_sparse
from .quantize import Nvfp4QuantizedTensor, dequantize_nvfp4_128x4_to_bf16


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


def _rows_per_batch(cu_seqlens_k: torch.Tensor, blk_kv: int) -> list[int]:
    seqlens = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).to("cpu").tolist()
    return [(int(v) + int(blk_kv) - 1) // int(blk_kv) for v in seqlens]


def _packed_row_to_batch_block(
    cu_seqlens_k: torch.Tensor,
    blk_kv: int,
) -> list[tuple[int, int]]:
    rows_per_batch = _rows_per_batch(cu_seqlens_k, int(blk_kv))
    max_rows = max(rows_per_batch, default=0)
    row_map: list[tuple[int, int]] = []
    for block_idx in range(max_rows):
        for batch_idx, rows in enumerate(rows_per_batch):
            if block_idx < rows:
                row_map.append((batch_idx, block_idx))
    return row_map


def _csr_to_q2k_indices(
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    blk_kv: int,
    topk: int,
) -> torch.Tensor:
    if k2q_row_ptr.dtype != torch.int32 or k2q_q_indices.dtype != torch.int32:
        raise TypeError("k2q_row_ptr and k2q_q_indices must be torch.int32")
    if k2q_row_ptr.ndim != 2 or k2q_q_indices.ndim != 2:
        raise ValueError("k2q_row_ptr and k2q_q_indices must be rank-2")
    head_kv = int(k2q_row_ptr.shape[0])
    total_q = int(cu_seqlens_q[-1].item())
    topk = int(topk)
    if topk <= 0:
        raise ValueError(f"topK must be positive, got {topk}")

    row_map = _packed_row_to_batch_block(cu_seqlens_k, int(blk_kv))
    if int(k2q_row_ptr.shape[1]) != len(row_map) + 1:
        raise ValueError(
            "k2q_row_ptr shape does not match cu_seqlens_k/blk_kv packed rows: "
            f"got {tuple(k2q_row_ptr.shape)}, expected second dim {len(row_map) + 1}"
        )

    cu_q_cpu = cu_seqlens_q.to("cpu", non_blocking=False).tolist()
    row_ptr_cpu = k2q_row_ptr.to("cpu", non_blocking=False)
    q_idx_cpu = k2q_q_indices.to("cpu", non_blocking=False)
    q2k = torch.full(
        (head_kv, total_q, topk),
        -1,
        dtype=torch.int32,
        device=k2q_row_ptr.device,
    )

    next_slot: dict[tuple[int, int], int] = {}
    seen: set[tuple[int, int, int]] = set()
    for h in range(head_kv):
        for row, (batch_idx, kv_block_idx) in enumerate(row_map):
            start = int(row_ptr_cpu[h, row].item())
            end = int(row_ptr_cpu[h, row + 1].item())
            q_start = int(cu_q_cpu[batch_idx])
            q_end = int(cu_q_cpu[batch_idx + 1])
            q_len = q_end - q_start
            for pos in range(start, end):
                q_val = int(q_idx_cpu[h, pos].item())
                if q_val < 0:
                    continue
                if q_start <= q_val < q_end:
                    q_global = q_val
                elif q_val < q_len:
                    q_global = q_start + q_val
                else:
                    continue
                key = (h, q_global, kv_block_idx)
                if key in seen:
                    continue
                seen.add(key)
                slot_key = (h, q_global)
                slot = next_slot.get(slot_key, 0)
                if slot >= topk:
                    continue
                q2k[h, q_global, slot] = int(kv_block_idx)
                next_slot[slot_key] = slot + 1
    return q2k.contiguous()


def _make_prefill_cache_and_metadata(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_k: int,
    blk_kv: int,
    *,
    page_table: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if page_table is None:
        if k.ndim != 3 or v.ndim != 3:
            raise ValueError("dense prefill k/v must have shape [total_k, Hkv, D]")
        if k.shape != v.shape:
            raise ValueError("k and v must have the same shape")
        k_cache = k.contiguous()
        v_cache = v.contiguous()
        batch = int(cu_seqlens_q.numel()) - 1
        max_seqlen_k = int(max_seqlen_k)
        logical = torch.arange(max_seqlen_k, device=q.device, dtype=torch.int64)
        cu_k = cu_seqlens_k.to(torch.int64)
        seq_lens = (cu_k[1:] - cu_k[:-1]).to(torch.int32).contiguous()
        valid = logical[None, :] < seq_lens.to(torch.int64)[:, None]
        req_to_token = (cu_k[:-1, None] + logical[None, :]).to(torch.int32)
        req_to_token = torch.where(valid, req_to_token, torch.zeros_like(req_to_token)).contiguous()
        slot_ids = torch.arange(batch, device=q.device, dtype=torch.int64)
    else:
        if k.ndim != 4 or v.ndim != 4:
            raise ValueError("paged prefill k/v must have shape [num_pages, Hkv, blk_kv, D]")
        if k.shape != v.shape:
            raise ValueError("k and v must have the same shape")
        if int(k.shape[2]) != int(blk_kv):
            raise ValueError("paged k/v page dimension must equal blk_kv")
        if page_table.dtype != torch.int32:
            raise TypeError("page_table must be torch.int32")
        k_cache = k.transpose(1, 2).reshape(k.shape[0] * k.shape[2], k.shape[1], k.shape[3]).contiguous()
        v_cache = v.transpose(1, 2).reshape(v.shape[0] * v.shape[2], v.shape[1], v.shape[3]).contiguous()
        batch = int(page_table.shape[0])
        if seqused_k is None:
            seq_lens = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).to(torch.int32).contiguous()
        else:
            seq_lens = seqused_k.to(torch.int32).contiguous()
        logical = torch.arange(int(max_seqlen_k), device=q.device, dtype=torch.int64)
        page = (logical // int(blk_kv)).clamp(max=int(page_table.shape[1]) - 1)
        offset = logical % int(blk_kv)
        physical_page = page_table[:, page].to(torch.int64)
        req_to_token = (physical_page * int(blk_kv) + offset[None, :]).to(torch.int32)
        valid = logical[None, :] < seq_lens.to(torch.int64)[:, None]
        req_to_token = torch.where(valid, req_to_token, torch.zeros_like(req_to_token)).contiguous()
        slot_ids = torch.arange(batch, device=q.device, dtype=torch.int64)

    q_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(torch.int32)
    prefix_lens = (seq_lens - q_lens).clamp_min(0).to(torch.int32).contiguous()
    return k_cache, v_cache, req_to_token, slot_ids, seq_lens, prefix_lens


def _select_prefill_block_size_q(q_heads: int, kv_heads: int) -> int:
    if kv_heads <= 0 or q_heads % kv_heads != 0:
        raise ValueError("number of query heads must be divisible by KV heads")
    gqa_group = q_heads // kv_heads
    limit = max(1, 128 // max(1, gqa_group))
    for candidate in (64, 32, 16, 8, 4, 2, 1):
        if candidate <= limit:
            return candidate
    return 1


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


def sparse_atten_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    topK: int,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    blk_kv: int = 128,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
    lse_temperature_scale: float = 1.0,
    return_temperature_lse: bool = False,
    partial_dtype: torch.dtype = torch.bfloat16,
    return_softmax_lse: bool = False,
    page_table: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    schedule: Optional[object] = None,
    usable_SM_count: int = -1,
    qk_dtype: Optional[torch.dtype] = None,
    pv_dtype: Optional[torch.dtype] = None,
):
    """Run the official MiniMax CSR sparse prefill API on Blackwell.

    The wrapper converts MiniMax's K-to-Q CSR selection to the Q-to-K block list
    consumed by the Blackwell Triton sparse prefill kernel.  The default output
    path is supported for BF16/FP16 tensors with head_dim=128.
    """

    del partial_dtype, usable_SM_count, qk_dtype, pv_dtype
    if schedule is not None:
        raise ValueError("schedule objects are not consumed by the Blackwell prefill wrapper")
    if return_temperature_lse or return_softmax_lse:
        raise ValueError("LSE return paths are not exposed by the Blackwell prefill wrapper")
    if float(lse_temperature_scale) != 1.0:
        raise ValueError("lse_temperature_scale values other than 1.0 are not supported")
    if not causal:
        raise ValueError("Blackwell sparse prefill wrapper currently supports causal=True")
    if q.ndim != 3:
        raise ValueError("q must have shape [total_q, Hq, D]")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("q must be BF16 or FP16")
    if q.shape[-1] != 128 or k.shape[-1] != 128 or v.shape[-1] != 128:
        raise ValueError("Blackwell sparse prefill wrapper currently supports head_dim=128")
    if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
        raise TypeError("cu_seqlens_q and cu_seqlens_k must be torch.int32")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")

    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("q, k, and v must have matching BF16/FP16 dtype")
    kv_heads = int(k.shape[1])
    q2k_indices = _csr_to_q2k_indices(
        k2q_row_ptr.contiguous(),
        k2q_q_indices.contiguous(),
        cu_seqlens_q.contiguous(),
        cu_seqlens_k.contiguous(),
        int(blk_kv),
        int(topK),
    )
    k_cache, v_cache, req_to_token, slot_ids, seq_lens, prefix_lens = _make_prefill_cache_and_metadata(
        q,
        k,
        v,
        cu_seqlens_q.contiguous(),
        cu_seqlens_k.contiguous(),
        int(max_seqlen_k),
        int(blk_kv),
        page_table=None if page_table is None else page_table.contiguous(),
        seqused_k=None if seqused_k is None else seqused_k.contiguous(),
    )
    block_size_q = _select_prefill_block_size_q(int(q.shape[1]), kv_heads)
    return flash_prefill_with_gqa_share_sparse(
        q.contiguous(),
        k_cache,
        v_cache,
        None,
        req_to_token,
        slot_ids,
        q2k_indices,
        block_size_q=block_size_q,
        block_size_k=int(blk_kv),
        cu_seqlens=cu_seqlens_q.contiguous(),
        seq_lens=seq_lens,
        prefix_lens=prefix_lens,
        max_seqlen_q=int(max_seqlen_q),
        sm_scale=softmax_scale,
    )


def _nvfp4_dequant_kv(
    packed: torch.Tensor,
    scale_128x4: torch.Tensor,
    global_scale: torch.Tensor,
) -> torch.Tensor:
    if packed.dtype is not torch.uint8:
        return packed
    original_shape = tuple(int(v) for v in packed.shape[:-1]) + (int(packed.shape[-1]) * 2,)
    rows = 1
    for dim in original_shape[:-1]:
        rows *= int(dim)
    qx = Nvfp4QuantizedTensor(
        data=packed,
        scale_128x4=scale_128x4,
        global_scale=global_scale,
        logical_scale_shape=(rows, int(original_shape[-1]) // 16),
        original_shape=original_shape,
    )
    return dequantize_nvfp4_128x4_to_bf16(qx)


def sparse_atten_nvfp4_kv_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_scale_128x4: torch.Tensor,
    v_scale_128x4: torch.Tensor,
    k_global_scale: torch.Tensor,
    v_global_scale: torch.Tensor,
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    topK: int,
    **kwargs,
):
    """NVFP4-KV official prefill API wrapper.

    The Blackwell compatibility path dequantizes NVFP4 K/V with the package's
    128x4 metadata helper, then dispatches to `sparse_atten_func`.
    """

    k_bf16 = _nvfp4_dequant_kv(k, k_scale_128x4, k_global_scale)
    v_bf16 = _nvfp4_dequant_kv(v, v_scale_128x4, v_global_scale)
    return sparse_atten_func(
        q,
        k_bf16,
        v_bf16,
        k2q_row_ptr,
        k2q_q_indices,
        topK,
        **kwargs,
    )


def _unpack_e2m1_fp4(packed: torch.Tensor) -> torch.Tensor:
    data = packed if packed.dtype is torch.uint8 else packed.view(torch.uint8)
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
        device=data.device,
    )
    out = torch.empty(tuple(data.shape[:-1]) + (int(data.shape[-1]) * 2,), dtype=torch.float32, device=data.device)
    out[..., 0::2] = lut[(data & 0x0F).long()]
    out[..., 1::2] = lut[(data >> 4).long()]
    return out


def _apply_public_fp4_scale(values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if scale.ndim == values.ndim and int(scale.shape[-1]) * 16 == int(values.shape[-1]):
        return values * scale.to(torch.float32).repeat_interleave(16, dim=-1)
    if scale.ndim == values.ndim - 1:
        return values * scale.to(torch.float32).unsqueeze(-1)
    return values


def fp4_indexer_block_scores(
    q_fp4: torch.Tensor,
    k_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_page_offsets: torch.Tensor,
    *,
    max_seqlen_q: int,
    max_seqlen_k: int,
    kv_indices: torch.Tensor,
    fp4_format: str,
    causal: bool = False,
    qo_offset: Optional[torch.Tensor] = None,
    scale_layout: str = "preordered_mma",
) -> torch.Tensor:
    """Correctness-first FP4 block-score API for Blackwell.

    Returns the official shape `[Hq, ceil(max_seqlen_k / 128), total_q]`.
    This fallback is intentionally simple and is used to keep the public API
    callable on Blackwell while optimized CUTE block scoring remains SM100-only.
    """

    del qo_offset
    if fp4_format.lower() not in {"nvfp4", "e2m1", "mxfp4"}:
        raise ValueError(f"unsupported fp4_format: {fp4_format}")
    if scale_layout not in {"public", "preordered_mma"}:
        raise ValueError(f"unsupported scale_layout: {scale_layout}")
    if q_fp4.ndim != 3:
        raise ValueError("q_fp4 must have shape [total_q, Hq, packed_D]")
    if k_fp4.ndim != 4:
        raise ValueError("k_fp4 must have shape [num_pages, Hkv, 128, packed_D]")
    total_q, hq, _ = q_fp4.shape
    num_pages, hkv, page_size, _ = k_fp4.shape
    if int(page_size) != 128:
        raise ValueError("fp4_indexer_block_scores currently expects 128-token pages")
    if hq % hkv != 0:
        raise ValueError("Hq must be divisible by Hkv")
    max_blocks = (int(max_seqlen_k) + 127) // 128
    scores = torch.full((hq, max_blocks, total_q), float("-inf"), dtype=torch.float32, device=q_fp4.device)

    q = _apply_public_fp4_scale(_unpack_e2m1_fp4(q_fp4), q_scale)
    k = _apply_public_fp4_scale(_unpack_e2m1_fp4(k_fp4), k_scale)

    # Native block-max QK kernel for the cubic scoring loop when available.
    # Same dequant numerics as the reference (bf16 compute); falls back to the
    # Python reference below in source-tree mode.
    if has_native_ops() and q_fp4.is_cuda:
        batch = int(cu_seqlens_q.numel()) - 1
        q_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(torch.int64)
        batch_of_q = torch.repeat_interleave(
            torch.arange(batch, device=q_fp4.device, dtype=torch.int32),
            q_lens,
        )
        return native_indexer_block_scores(
            q.to(torch.bfloat16),
            k.to(torch.bfloat16),
            batch_of_q,
            cu_seqlens_q.to(torch.int32),
            cu_seqlens_k.to(torch.int32),
            cu_page_offsets.to(torch.int32),
            kv_indices.to(torch.int32),
            max_blocks=max_blocks,
            causal=causal,
        )

    gqa = hq // hkv
    batch = int(cu_seqlens_q.numel()) - 1
    cu_q = cu_seqlens_q.to("cpu", non_blocking=False).tolist()
    cu_k = cu_seqlens_k.to("cpu", non_blocking=False).tolist()
    cu_pages = cu_page_offsets.to("cpu", non_blocking=False).tolist()
    kv_idx_cpu = kv_indices.to("cpu", non_blocking=False).tolist()
    for b in range(batch):
        q_start, q_end = int(cu_q[b]), int(cu_q[b + 1])
        k_len = int(cu_k[b + 1]) - int(cu_k[b])
        page_start, page_end = int(cu_pages[b]), int(cu_pages[b + 1])
        for logical_block, page_pos in enumerate(range(page_start, page_end)):
            if logical_block >= max_blocks:
                break
            physical_page = int(kv_idx_cpu[page_pos])
            if physical_page < 0 or physical_page >= num_pages:
                continue
            valid_tokens = min(128, max(0, k_len - logical_block * 128))
            if valid_tokens <= 0:
                continue
            for qi in range(q_start, q_end):
                local_q = qi - q_start
                if causal and logical_block * 128 > local_q:
                    continue
                for qh in range(hq):
                    kh = qh // gqa
                    key_block = k[physical_page, kh, :valid_tokens]
                    if causal:
                        visible = min(valid_tokens, max(0, local_q - logical_block * 128 + 1))
                        if visible <= 0:
                            continue
                        key_block = key_block[:visible]
                    block_scores = torch.matmul(key_block, q[qi, qh])
                    scores[qh, logical_block, qi] = block_scores.max()
    return scores


__all__ = [
    "sparse_atten_func",
    "sparse_atten_nvfp4_kv_func",
    "sparse_decode_atten_func",
    "SparseDecodePagedAttentionWrapper",
    "fp4_indexer_block_scores",
]
