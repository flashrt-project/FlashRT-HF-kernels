"""FlashRT QKV epilogue training reference API."""
from __future__ import annotations
import torch
try:
    from ._ops import ops
except Exception:  # source-tree tests before kernel-builder creates _ops.py
    class _SourceOpsFallback:
        def _flashrt_training_package_marker(self, x):
            return x
    ops = _SourceOpsFallback()

def _rotate_half(x):
    half=x.shape[-1]//2; return torch.cat((-x[...,half:], x[...,:half]), dim=-1)
def _rope(x, cos, sin):
    compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    c=cos[:,None,:,:].to(compute_dtype) if cos.dim()==3 else cos[None,None,:,:].to(compute_dtype)
    s=sin[:,None,:,:].to(compute_dtype) if sin.dim()==3 else sin[None,None,:,:].to(compute_dtype)
    xf = x.to(compute_dtype)
    return (xf*c + _rotate_half(xf)*s).to(x.dtype)
def qkv_rope_reference(x: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, q_heads: int, kv_heads: int, head_dim: int):
    b,t,hid=x.shape
    q=(x@wq.t()).view(b,t,int(q_heads),int(head_dim)).transpose(1,2).contiguous(); k=(x@wk.t()).view(b,t,int(kv_heads),int(head_dim)).transpose(1,2).contiguous(); v=(x@wv.t()).view(b,t,int(kv_heads),int(head_dim)).transpose(1,2).contiguous()
    return _rope(q,cos,sin), _rope(k,cos,sin), v
def backend_marker(x: torch.Tensor) -> torch.Tensor:
    return ops._flashrt_training_package_marker(x)
__all__=["qkv_rope_reference","backend_marker"]
