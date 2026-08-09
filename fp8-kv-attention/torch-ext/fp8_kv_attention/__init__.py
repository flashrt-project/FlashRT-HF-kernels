"""FlashRT BF16-Q + FP8-KV XQA attention kernels."""

from __future__ import annotations

from typing import Optional

import torch

from ._ops import add_op_namespace_prefix, ops


PAGE_SIZE = 128
SUPPORTED_CONFIGS = (
    (24, 4, 256),
    (16, 2, 256),
    (32, 8, 128),
    (32, 16, 128),
    (16, 8, 128),
)


@torch.library.register_fake(add_op_namespace_prefix("xqa_bf16_fp8kv"))
def _xqa_bf16_fp8kv_fake(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    mask: torch.Tensor,
    out: torch.Tensor,
    semaphores: torch.Tensor,
    scratch: torch.Tensor,
    max_seq_len: int = 0,
    q_scale: float = 1.0,
    kv_scale: float = 1.0,
    enable_pdl: bool = True,
    sm_count: int = 0,
    k_stride_page: int = 0,
    k_stride_token: int = 0,
    k_stride_head: int = 0,
) -> None:
    if q.dim() == 3:
        q_seq = q.shape[0]
        q_heads, head_dim = q.shape[1:]
    elif q.dim() == 5:
        q_seq = q.shape[2]
        if q.shape[:2] != (1, 1):
            raise RuntimeError("rank-5 q must begin with (1,1)")
        q_heads, head_dim = q.shape[3:]
    else:
        raise RuntimeError("q must have rank 3 or 5")
    if out.shape != q.shape:
        raise RuntimeError("q/out shape mismatch")
    if k_cache.dim() != 4 or k_cache.shape[1] != PAGE_SIZE or k_cache.shape[3] != head_dim:
        raise RuntimeError("k_cache shape does not match q")
    if v_cache.shape != k_cache.shape:
        raise RuntimeError("v_cache shape mismatch")
    config = (q_heads, k_cache.shape[2], head_dim)
    if config not in SUPPORTED_CONFIGS:
        raise RuntimeError(f"unsupported XQA configuration: {config}")
    if mask.numel() < q_seq * ((q_seq + 31) // 32):
        raise RuntimeError("mask is too small")
    return None


def causal_spec_mask(q_seq: int, *, device: torch.device | str = "cuda", dtype: torch.dtype = torch.int32) -> torch.Tensor:
    """Return the packed lower-triangular mask expected by the XQA kernel."""

    q_seq = int(q_seq)
    words = (q_seq + 31) // 32
    rows = torch.zeros((q_seq, words), dtype=torch.int32)
    for i in range(q_seq):
        upto = i + 1
        full = upto // 32
        rem = upto % 32
        if full:
            rows[i, :full] = -1
        if rem:
            rows[i, full] = (1 << rem) - 1
    return rows.to(device=device, dtype=dtype)


def default_page_table(num_pages: int, *, device: torch.device | str = "cuda") -> torch.Tensor:
    """Return a contiguous one-batch page table."""

    return torch.arange(int(num_pages), device=device, dtype=torch.int32).view(1, int(num_pages))


def allocate_workspace(
    *,
    q_seq: int,
    num_q_heads: int = 24,
    num_kv_heads: int = 4,
    device: torch.device | str = "cuda",
    scratch_mb: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate semaphores and scratch tensors for static-buffer runtimes."""

    if num_q_heads % num_kv_heads:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    sem_count = num_kv_heads * (
        ((int(q_seq) * (num_q_heads // num_kv_heads)) + 31) // 32
    )
    semaphores = torch.zeros(max(256, sem_count), device=device, dtype=torch.int32)
    scratch = torch.empty(max(1, int(scratch_mb)) << 20, device=device, dtype=torch.uint8)
    return semaphores, scratch


def xqa_bf16_fp8kv(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    *,
    out: Optional[torch.Tensor] = None,
    semaphores: Optional[torch.Tensor] = None,
    scratch: Optional[torch.Tensor] = None,
    max_seq_len: int = 0,
    q_scale: float = 1.0,
    kv_scale: float = 1.0,
    enable_pdl: bool = True,
    sm_count: int = 0,
    k_stride_page: int = 0,
    k_stride_token: int = 0,
    k_stride_head: int = 0,
) -> torch.Tensor:
    """Run a validated BF16-query / FP8-KV paged XQA configuration."""

    if out is None:
        out = torch.empty_like(q)
    if page_table is None:
        page_table = default_page_table(k_cache.shape[0], device=q.device)
    if seq_lens is None:
        seq_lens = torch.tensor([[k_cache.shape[0] * PAGE_SIZE]], device=q.device, dtype=torch.int32)
    if mask is None:
        q_seq = q.shape[0] if q.dim() == 3 else q.shape[2]
        mask = causal_spec_mask(int(q_seq), device=q.device, dtype=torch.int32)
    if semaphores is None or scratch is None:
        q_seq = q.shape[0] if q.dim() == 3 else q.shape[2]
        q_heads = q.shape[1] if q.dim() == 3 else q.shape[3]
        semaphores_new, scratch_new = allocate_workspace(
            q_seq=int(q_seq),
            num_q_heads=int(q_heads),
            num_kv_heads=int(k_cache.shape[2]),
            device=q.device,
        )
        if semaphores is None:
            semaphores = semaphores_new
        if scratch is None:
            scratch = scratch_new
    ops.xqa_bf16_fp8kv(
        q,
        k_cache,
        v_cache,
        page_table,
        seq_lens,
        mask,
        out,
        semaphores,
        scratch,
        int(max_seq_len),
        float(q_scale),
        float(kv_scale),
        bool(enable_pdl),
        int(sm_count),
        int(k_stride_page),
        int(k_stride_token),
        int(k_stride_head),
    )
    return out


__all__ = [
    "PAGE_SIZE",
    "SUPPORTED_CONFIGS",
    "allocate_workspace",
    "causal_spec_mask",
    "default_page_table",
    "xqa_bf16_fp8kv",
]
