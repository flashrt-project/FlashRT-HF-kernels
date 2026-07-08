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
    while freq.dim() < x.dim(): freq = freq.unsqueeze(unsqueeze_dim)
    return freq
def apply_rope_train(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1):
    c_q=_align(cos,q,int(unsqueeze_dim)); s_q=_align(sin,q,int(unsqueeze_dim)); c_k=_align(cos,k,int(unsqueeze_dim)); s_k=_align(sin,k,int(unsqueeze_dim))
    qd = torch.float64 if q.dtype == torch.float64 else torch.float32
    kd = torch.float64 if k.dtype == torch.float64 else torch.float32
    return (q.to(qd)*c_q.to(qd)+rotate_half(q.to(qd))*s_q.to(qd)).to(q.dtype), (k.to(kd)*c_k.to(kd)+rotate_half(k.to(kd))*s_k.to(kd)).to(k.dtype)
def apply_rope_backward_reference(dq: torch.Tensor, dk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1):
    return apply_rope_train(dq, dk, cos, -sin, unsqueeze_dim)
def backend_marker(x: torch.Tensor) -> torch.Tensor:
    return ops._flashrt_training_package_marker(x)
__all__=["rotate_half","apply_rope_train","apply_rope_backward_reference","backend_marker"]
