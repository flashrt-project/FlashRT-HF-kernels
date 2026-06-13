# SPDX-License-Identifier: Apache-2.0
"""Sparse-index helpers compatible with MiniMaxAI/msa public names."""

from __future__ import annotations

from typing import Optional

import torch


def _validate_cu_seqlens(cu_seqlens: torch.Tensor, *, name: str) -> None:
    if cu_seqlens.dtype != torch.int32:
        raise TypeError(f"{name} must be torch.int32, got {cu_seqlens.dtype}")
    if cu_seqlens.ndim != 1:
        raise ValueError(f"{name} must be rank-1, got shape {tuple(cu_seqlens.shape)}")
    if cu_seqlens.numel() < 1:
        raise ValueError(f"{name} must have at least one element")
    if not cu_seqlens.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _rows_per_batch(cu_seqlens_k: torch.Tensor, kv_block_size: int) -> torch.Tensor:
    seqlens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    return (seqlens_k + kv_block_size - 1) // kv_block_size


def _build_packed_row_map(rows_per_batch: torch.Tensor) -> tuple[torch.Tensor, int]:
    rows_per_batch_cpu = rows_per_batch.to("cpu", non_blocking=False).tolist()
    batch = len(rows_per_batch_cpu)
    max_rows = max(rows_per_batch_cpu, default=0)
    row_dtype = (
        torch.int32
        if sum(rows_per_batch_cpu) < torch.iinfo(torch.int32).max
        else torch.int64
    )
    row_map_cpu = torch.full((batch, max_rows), -1, dtype=row_dtype)
    row_linear = 0
    for kv_block_idx in range(max_rows):
        for batch_idx, row_count in enumerate(rows_per_batch_cpu):
            if kv_block_idx < row_count:
                row_map_cpu[batch_idx, kv_block_idx] = row_linear
                row_linear += 1
    return row_map_cpu.to(rows_per_batch.device), row_linear


def build_k2q_csr_torch_reference(
    q2k_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    kv_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if kv_block_size <= 0:
        raise ValueError(f"kv_block_size must be > 0, got {kv_block_size}")
    if q2k_indices.dtype != torch.int32:
        raise TypeError(f"q2k_indices must be torch.int32, got {q2k_indices.dtype}")
    if q2k_indices.ndim != 3:
        raise ValueError(
            "q2k_indices must have shape [head_kv, total_q, topK], "
            f"got {tuple(q2k_indices.shape)}"
        )
    _validate_cu_seqlens(cu_seqlens_q, name="cu_seqlens_q")
    _validate_cu_seqlens(cu_seqlens_k, name="cu_seqlens_k")
    if cu_seqlens_q.shape != cu_seqlens_k.shape:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must have the same shape [B + 1]")
    if q2k_indices.device != cu_seqlens_q.device or q2k_indices.device != cu_seqlens_k.device:
        raise ValueError("q2k_indices, cu_seqlens_q, and cu_seqlens_k must be on the same device")

    head_kv, total_q, topk = q2k_indices.shape
    if total_q != int(cu_seqlens_q[-1].item()):
        raise ValueError(
            f"q2k_indices.shape[1] ({total_q}) must equal cu_seqlens_q[-1] "
            f"({int(cu_seqlens_q[-1].item())})"
        )

    rows_per_batch = _rows_per_batch(cu_seqlens_k, kv_block_size)
    row_map, total_rows = _build_packed_row_map(rows_per_batch)
    nnz_upper_bound = total_q * topk

    k2q_row_ptr = torch.zeros((head_kv, total_rows + 1), dtype=torch.int32, device=q2k_indices.device)
    k2q_q_indices = torch.full(
        (head_kv, nnz_upper_bound), -1, dtype=torch.int32, device=q2k_indices.device
    )
    if total_rows == 0 or total_q == 0 or topk == 0:
        return k2q_row_ptr, k2q_q_indices

    counts = torch.zeros((head_kv, total_rows), dtype=torch.int32, device=q2k_indices.device)
    total_entries = total_q * topk
    row_dtype = torch.int32 if total_rows < torch.iinfo(torch.int32).max else torch.int64
    row_all = torch.empty((head_kv, total_entries), dtype=row_dtype, device=q2k_indices.device)
    q_all = torch.empty((head_kv, total_entries), dtype=torch.int32, device=q2k_indices.device)
    valid_all = torch.empty((head_kv, total_entries), dtype=torch.bool, device=q2k_indices.device)

    for batch_idx in range(cu_seqlens_q.numel() - 1):
        q_start = int(cu_seqlens_q[batch_idx].item())
        q_end = int(cu_seqlens_q[batch_idx + 1].item())
        if q_end <= q_start:
            continue
        q_local = torch.arange(q_start, q_end, dtype=torch.int32, device=q2k_indices.device)
        q_local = q_local[:, None].expand(q_end - q_start, topk).reshape(-1)
        kv_local = q2k_indices[:, q_start:q_end, :].reshape(head_kv, -1)
        valid = kv_local >= 0
        safe_kv = kv_local.clamp_min(0).long()
        rows = row_map[batch_idx, safe_kv].to(row_dtype)
        start = q_start * topk
        end = q_end * topk
        row_all[:, start:end] = rows
        q_all[:, start:end] = q_local[None, :].expand(head_kv, -1)
        valid_all[:, start:end] = valid & (rows >= 0)
        counts.scatter_add_(1, rows.clamp_min(0).long(), valid_all[:, start:end].to(torch.int32))

    k2q_row_ptr[:, 1:] = counts.cumsum(dim=1)
    offsets = torch.zeros_like(counts)
    offsets[:, 1:] = counts[:, :-1].cumsum(dim=1)
    cursor = torch.zeros_like(counts)
    for h in range(head_kv):
        for entry in range(total_entries):
            if not bool(valid_all[h, entry]):
                continue
            row = int(row_all[h, entry].item())
            pos = int(offsets[h, row].item() + cursor[h, row].item())
            k2q_q_indices[h, pos] = q_all[h, entry]
            cursor[h, row] += 1

    return k2q_row_ptr, k2q_q_indices


def build_k2q_csr(
    q2k_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    kv_block_size: int,
    *,
    total_k: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    max_seqlen_q: Optional[int] = None,
    total_rows: Optional[int] = None,
    qhead_per_kv: int = 1,
    return_schedule: bool = False,
):
    if return_schedule:
        raise NotImplementedError(
            "Blackwell build_k2q_csr currently returns CSR tensors only; "
            "SM100 schedule objects are not part of this package."
        )
    if total_k is None:
        raise ValueError("build_k2q_csr requires total_k from k.shape[0]")
    return build_k2q_csr_torch_reference(
        q2k_indices.contiguous(),
        cu_seqlens_q.contiguous(),
        cu_seqlens_k.contiguous(),
        int(kv_block_size),
    )


class SparseK2qCsrBuilderSm100:
    """Compatibility placeholder for the upstream SM100 class name.

    The name is exported so code can feature-detect it, but schedule-producing
    SM100 behavior is intentionally not claimed for this Blackwell package.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def build(self, *args, **kwargs):
        return build_k2q_csr(*args, **kwargs)


__all__ = [
    "build_k2q_csr",
    "build_k2q_csr_torch_reference",
    "SparseK2qCsrBuilderSm100",
]
