"""Static-buffer FlashAttention-2 runtime operators from FlashRT."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch

from ._ops import add_op_namespace_prefix, ops


SUPPORTED_HEAD_DIMS = (64, 96, 128, 256)
SPLIT_HEAD_DIMS = (96, 128, 256)


@dataclass(frozen=True)
class FA2Workspace:
    """Preallocated split-KV workspace for one static attention shape."""

    softmax_lse_accum: torch.Tensor
    out_accum: torch.Tensor
    num_sms: int
    num_splits: int


def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def recommended_num_splits(
    batch: int,
    seqlen_q: int,
    seqlen_k: int,
    heads_q: int,
    head_dim: int,
    num_sms: int,
) -> int:
    """Return the exact split count selected by the FlashRT FA2 heuristic."""

    values = (batch, seqlen_q, seqlen_k, heads_q, head_dim, num_sms)
    if any(int(v) <= 0 for v in values):
        raise ValueError("all shape values and num_sms must be positive")
    if int(head_dim) not in SUPPORTED_HEAD_DIMS:
        raise ValueError(f"head_dim must be one of {SUPPORTED_HEAD_DIMS}")
    block_n = 256 if head_dim <= 64 else (128 if head_dim <= 128 else 64)
    n_blocks = _ceildiv(seqlen_k, block_n)
    m_blocks = _ceildiv(seqlen_q, 64)
    blocks = batch * heads_q * m_blocks
    effective_sms = num_sms * 2
    if blocks >= 0.8 * effective_sms:
        return 1
    max_splits = min(128, effective_sms, n_blocks)
    efficiencies = [0.0] * (max_splits + 1)
    best = 0.0
    for split in range(1, max_splits + 1):
        eligible = split == 1 or _ceildiv(n_blocks, split) != _ceildiv(n_blocks, split - 1)
        if not eligible:
            continue
        waves = blocks * split / effective_sms
        efficiencies[split] = waves / math.ceil(waves)
        best = max(best, efficiencies[split])
    for split in range(1, max_splits + 1):
        eligible = split == 1 or _ceildiv(n_blocks, split) != _ceildiv(n_blocks, split - 1)
        if eligible and efficiencies[split] >= 0.85 * best:
            return split
    return 1


def allocate_workspace(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    num_sms: Optional[int] = None,
) -> Optional[FA2Workspace]:
    """Allocate the exact split-KV workspace selected for ``q`` and ``k``.

    Returns ``None`` when the heuristic selects the no-split path. Allocate
    once during runtime setup; never call this helper inside a captured loop.
    """

    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("q and k must have shape (B, S, H, D)")
    if q.shape[-1] not in SPLIT_HEAD_DIMS:
        return None
    if num_sms is None:
        num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    splits = recommended_num_splits(
        q.shape[0], q.shape[1], k.shape[1], q.shape[2], q.shape[3], num_sms
    )
    if splits == 1:
        return None
    lse = torch.empty(
        (splits, q.shape[0], q.shape[2], q.shape[1]),
        device=q.device,
        dtype=torch.float32,
    )
    out = torch.empty(
        (splits, q.shape[0], q.shape[2], q.shape[1], q.shape[3]),
        device=q.device,
        dtype=torch.float32,
    )
    return FA2Workspace(lse, out, int(num_sms), int(splits))


def allocate_outputs(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate output and LSE tensors for a static ``(B,S,H,D)`` query."""

    if q.ndim != 4:
        raise ValueError("q must have shape (B, S, H, D)")
    out = torch.empty_strided(q.shape, q.stride(), device=q.device, dtype=q.dtype)
    lse = torch.empty(
        (q.shape[0], q.shape[2], q.shape[1]),
        device=q.device,
        dtype=torch.float32,
    )
    return out, lse


def _workspace_args(
    workspace: Optional[FA2Workspace],
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
    if workspace is None:
        return None, None, 0
    return workspace.softmax_lse_accum, workspace.out_accum, int(workspace.num_sms)


@torch.library.register_fake(add_op_namespace_prefix("forward_static"))
def _forward_static_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    softmax_lse_accum: Optional[torch.Tensor],
    out_accum: Optional[torch.Tensor],
    softmax_scale: float,
    causal: bool = False,
    num_sms: int = 0,
) -> None:
    del k, v, softmax_scale, causal, num_sms
    if q.ndim != 4 or out.shape != q.shape:
        raise RuntimeError("q/out must have matching (B, S, H, D) shapes")
    if softmax_lse.shape != (q.shape[0], q.shape[2], q.shape[1]):
        raise RuntimeError("softmax_lse must have shape (B, H, S)")
    if (softmax_lse_accum is None) != (out_accum is None):
        raise RuntimeError("split-KV workspace tensors must be both set or both None")
    return None


@torch.library.register_fake(add_op_namespace_prefix("forward_seqused_static"))
def _forward_seqused_static_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seqused_k: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    softmax_lse_accum: Optional[torch.Tensor],
    out_accum: Optional[torch.Tensor],
    softmax_scale: float,
    num_sms: int = 0,
) -> None:
    del seqused_k
    return _forward_static_fake(
        q,
        k,
        v,
        out,
        softmax_lse,
        softmax_lse_accum,
        out_accum,
        softmax_scale,
        False,
        num_sms,
    )


def forward_static(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    workspace: Optional[FA2Workspace] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """Run allocation-free FA2 forward into caller-owned static buffers."""

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    lse_accum, out_accum, num_sms = _workspace_args(workspace)
    ops.forward_static(
        q,
        k,
        v,
        out,
        softmax_lse,
        lse_accum,
        out_accum,
        float(softmax_scale),
        bool(causal),
        int(num_sms),
    )
    return out


def forward_seqused_static(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seqused_k: torch.Tensor,
    *,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    workspace: Optional[FA2Workspace] = None,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Run BF16 FA2 with device-resident per-batch K/V lengths.

    Values in ``seqused_k`` must be in ``[1, k.shape[1]]``. When split-KV is
    enabled, the LSE workspace is reset to ``-inf`` on the current stream; that
    reset is captured together with the kernel by CUDA Graphs.
    """

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    lse_accum, out_accum, num_sms = _workspace_args(workspace)
    if lse_accum is not None:
        lse_accum.fill_(-torch.inf)
    ops.forward_seqused_static(
        q,
        k,
        v,
        seqused_k,
        out,
        softmax_lse,
        lse_accum,
        out_accum,
        float(softmax_scale),
        int(num_sms),
    )
    return out


def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    use_split_kv: bool = True,
) -> torch.Tensor:
    """Convenience API that allocates outputs and optional split-KV workspace."""

    out, lse = allocate_outputs(q)
    workspace = allocate_workspace(q, k) if use_split_kv else None
    return forward_static(
        q,
        k,
        v,
        out=out,
        softmax_lse=lse,
        workspace=workspace,
        softmax_scale=softmax_scale,
        causal=causal,
    )


__all__ = [
    "FA2Workspace",
    "SPLIT_HEAD_DIMS",
    "SUPPORTED_HEAD_DIMS",
    "allocate_outputs",
    "allocate_workspace",
    "forward",
    "forward_seqused_static",
    "forward_static",
    "recommended_num_splits",
]
