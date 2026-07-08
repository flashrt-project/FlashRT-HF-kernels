"""FlashRT huge-vocab CE training reference API."""
from __future__ import annotations
import torch
import torch.nn.functional as F
try:
    from ._ops import ops
except Exception:  # source-tree tests before kernel-builder creates _ops.py
    class _SourceOpsFallback:
        def _flashrt_training_package_marker(self, x):
            return x
    ops = _SourceOpsFallback()

def vocab_ce_loss(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, z_loss_weight: float = 0.0, ignore_index: int = -100) -> torch.Tensor:
    compute_dtype = torch.float64 if hidden.dtype == torch.float64 or weight.dtype == torch.float64 else torch.float32
    logits = (hidden.to(compute_dtype) @ weight.to(compute_dtype).t()); valid = labels != int(ignore_index); n_valid = valid.sum().clamp(min=1)
    loss = F.cross_entropy(logits, labels, ignore_index=int(ignore_index), reduction="sum") / n_valid
    if float(z_loss_weight) > 0:
        lse = torch.logsumexp(logits, dim=-1); loss = loss + float(z_loss_weight) * (lse.square() * valid.to(lse.dtype)).sum() / n_valid
    return loss

def vocab_ce_fwd(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, z_loss_weight: float = 0.0, ignore_index: int = -100):
    compute_dtype = torch.float64 if hidden.dtype == torch.float64 or weight.dtype == torch.float64 else torch.float32
    logits = (hidden.to(compute_dtype) @ weight.to(compute_dtype).t()); return vocab_ce_loss(hidden, weight, labels, z_loss_weight, ignore_index), torch.logsumexp(logits, dim=-1), logits.max(dim=-1).values

def backend_marker(x: torch.Tensor) -> torch.Tensor:
    return ops._flashrt_training_package_marker(x)
__all__=["vocab_ce_loss","vocab_ce_fwd","backend_marker"]
