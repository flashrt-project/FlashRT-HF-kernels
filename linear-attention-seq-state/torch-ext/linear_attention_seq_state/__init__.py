"""FlashRT sequential state-scan kernels for linear attention."""

from __future__ import annotations

from typing import Optional

import torch

from ._ops import add_op_namespace_prefix, ops


@torch.library.register_fake(add_op_namespace_prefix("gated_delta_recurrent_seq_bf16"))
def _gated_delta_recurrent_seq_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    out: torch.Tensor,
    use_qk_l2norm: bool = False,
) -> None:
    if q.dim() != 3 or k.shape != q.shape or v.shape != q.shape:
        raise RuntimeError("q/k/v must have shape (S,H,D)")
    if g.shape != (q.shape[0], q.shape[1]) or beta.shape != g.shape:
        raise RuntimeError("g/beta must have shape (S,H)")
    if state.shape != (q.shape[1], q.shape[2], q.shape[2]) or out.shape != q.shape:
        raise RuntimeError("state/out shape mismatch")
    return None


def gated_delta_recurrent_seq_bf16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    use_qk_l2norm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scan a full `(S,H,128)` Gated DeltaNet sequence in one launch.

    `state` is updated in place with the final `(H,128,128)` state.
    """

    if out is None:
        out = torch.empty_like(q)
    ops.gated_delta_recurrent_seq_bf16(q, k, v, g, beta, state, out, bool(use_qk_l2norm))
    return out, state


__all__ = ["gated_delta_recurrent_seq_bf16"]
