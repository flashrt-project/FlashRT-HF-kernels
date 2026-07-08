"""FlashRT SigLIP forward-fusion reference API."""
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

def siglip_residual_layernorm_fwd(x: torch.Tensor, residual: torch.Tensor | None, weight: torch.Tensor, bias: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    y = x if residual is None else x + residual
    return F.layer_norm(y.float(), (y.shape[-1],), weight.float(), None if bias is None else bias.float(), float(eps)).to(x.dtype)
def siglip_gelu_fwd(x: torch.Tensor, bias: torch.Tensor | None = None, approximate: str = "tanh") -> torch.Tensor:
    y = x if bias is None else x + bias
    return F.gelu(y.float(), approximate=approximate).to(x.dtype)
def use_fused_siglip_path() -> bool:
    return not torch.is_grad_enabled()
def backend_marker(x: torch.Tensor) -> torch.Tensor:
    return ops._flashrt_training_package_marker(x)
__all__=["siglip_residual_layernorm_fwd","siglip_gelu_fwd","use_fused_siglip_path","backend_marker"]
