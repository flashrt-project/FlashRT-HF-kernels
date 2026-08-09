"""FlashRT RoPE training reference API."""
from __future__ import annotations
import torch
try:
    from ._ops import ops
except Exception:  # source-tree tests before kernel-builder creates _ops.py
    class _SourceOpsFallback:
        def _flashrt_training_package_marker(self, x):
            return x
    ops = _SourceOpsFallback()

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half=x.shape[-1]//2; return torch.cat((-x[...,half:], x[...,:half]), dim=-1)
def _align(freq: torch.Tensor, x: torch.Tensor, unsqueeze_dim: int) -> torch.Tensor:
    if freq.dim() == 2:
        freq = freq.reshape((1,) * (x.dim() - 2) + freq.shape)
        return freq
    while freq.dim() < x.dim(): freq = freq.unsqueeze(unsqueeze_dim)
    return freq
def _rope_one(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, qd: torch.dtype, unsqueeze_dim: int) -> torch.Tensor:
    xf = x.to(qd)
    return (xf * _align(cos, x, unsqueeze_dim).to(qd) + rotate_half(xf) * _align(sin, x, unsqueeze_dim).to(qd)).to(x.dtype)
def apply_rope_train(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1):
    qd = torch.float64 if q.dtype == torch.float64 else torch.float32
    return _rope_one(q, cos, sin, qd, unsqueeze_dim), _rope_one(k, cos, sin, qd, unsqueeze_dim)
def apply_rope_backward_reference(dq: torch.Tensor, dk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1):
    return apply_rope_train(dq, dk, cos, -sin, unsqueeze_dim)
def backend_marker(x: torch.Tensor) -> torch.Tensor:
    return ops._flashrt_training_package_marker(x)
__all__=["rotate_half","apply_rope_train","apply_rope_backward_reference","backend_marker"]
