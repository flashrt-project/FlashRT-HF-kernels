"""FlashRT speculative decoding helper kernels."""

from __future__ import annotations

from typing import Optional

import torch

from ._ops import add_op_namespace_prefix, ops


@torch.library.register_fake(add_op_namespace_prefix("argmax_bf16"))
def _argmax_bf16_fake(logits: torch.Tensor, argmax_out: torch.Tensor) -> None:
    if logits.dim() != 2 or argmax_out.shape != (logits.shape[0],):
        raise RuntimeError("argmax_bf16 expects logits (rows,vocab), argmax_out (rows,)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("accept_greedy_bf16"))
def _accept_greedy_bf16_fake(
    logits: torch.Tensor,
    drafts: torch.Tensor,
    argmax_out: torch.Tensor,
    accept_n: torch.Tensor,
    spec_k: int,
) -> None:
    if logits.dim() != 2 or argmax_out.shape != (logits.shape[0],):
        raise RuntimeError("accept_greedy_bf16 expects logits (rows,vocab), argmax_out (rows,)")
    if drafts.dim() != 1 or drafts.numel() < spec_k or accept_n.numel() < 1:
        raise RuntimeError("drafts/accept_n shape mismatch")
    return None


@torch.library.register_fake(add_op_namespace_prefix("accept_partitioned_bf16"))
def _accept_partitioned_bf16_fake(
    logits: torch.Tensor,
    drafts: torch.Tensor,
    argmax_out: torch.Tensor,
    accept_n: torch.Tensor,
    partial_vals: torch.Tensor,
    partial_idx: torch.Tensor,
    spec_k: int,
    parts: int,
) -> None:
    if partial_vals.shape != (logits.shape[0], parts) or partial_idx.shape != (logits.shape[0], parts):
        raise RuntimeError("partial buffers must have shape (rows, parts)")
    return _accept_greedy_bf16_fake(logits, drafts, argmax_out, accept_n, spec_k)


def argmax_bf16(logits: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if out is None:
        out = torch.empty((logits.shape[0],), device=logits.device, dtype=torch.int64)
    ops.argmax_bf16(logits, out)
    return out


def accept_greedy_bf16(
    logits: torch.Tensor,
    drafts: torch.Tensor,
    spec_k: int,
    *,
    argmax_out: Optional[torch.Tensor] = None,
    accept_n: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if argmax_out is None:
        argmax_out = torch.empty((logits.shape[0],), device=logits.device, dtype=torch.int64)
    if accept_n is None:
        accept_n = torch.empty((1,), device=logits.device, dtype=torch.int32)
    ops.accept_greedy_bf16(logits, drafts, argmax_out, accept_n, int(spec_k))
    return argmax_out, accept_n


def accept_partitioned_bf16(
    logits: torch.Tensor,
    drafts: torch.Tensor,
    spec_k: int,
    parts: Optional[int] = None,
    *,
    argmax_out: Optional[torch.Tensor] = None,
    accept_n: Optional[torch.Tensor] = None,
    partial_vals: Optional[torch.Tensor] = None,
    partial_idx: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if parts is None:
        vocab = int(logits.shape[1])
        parts = 32 if vocab >= 131072 else (16 if vocab >= 65536 else 8)
    if argmax_out is None:
        argmax_out = torch.empty((logits.shape[0],), device=logits.device, dtype=torch.int64)
    if accept_n is None:
        accept_n = torch.empty((1,), device=logits.device, dtype=torch.int32)
    if partial_vals is None:
        partial_vals = torch.empty((logits.shape[0], parts), device=logits.device, dtype=torch.float32)
    if partial_idx is None:
        partial_idx = torch.empty((logits.shape[0], parts), device=logits.device, dtype=torch.int32)
    ops.accept_partitioned_bf16(
        logits, drafts, argmax_out, accept_n, partial_vals, partial_idx, int(spec_k), int(parts)
    )
    return argmax_out, accept_n


__all__ = [
    "argmax_bf16",
    "accept_greedy_bf16",
    "accept_partitioned_bf16",
]
