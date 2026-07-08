"""FlashRT AdaRMS training reference API."""
from __future__ import annotations
import torch
from torch import nn
try:
    from ._ops import ops
except Exception:  # source-tree tests before kernel-builder creates _ops.py
    class _SourceOpsFallback:
        def _flashrt_training_package_marker(self, x):
            return x
    ops = _SourceOpsFallback()

def _broadcast_modulation(x: torch.Tensor, modulation: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3 and modulation.dim() == 2:
        return modulation.unsqueeze(1)
    return modulation

def _compute_dtype(x: torch.Tensor) -> torch.dtype:
    return torch.float64 if x.dtype == torch.float64 else torch.float32

def _rms_norm(x: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    compute_dtype = _compute_dtype(x)
    xf = x.to(compute_dtype)
    rstd = torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + float(eps))
    return xf * rstd, rstd.squeeze(-1)

def adarms_forward(x: torch.Tensor, modulation_or_weight: torch.Tensor, eps: float = 1e-6, adaptive: bool = True):
    normed, rstd = _rms_norm(x, eps)
    if adaptive:
        modulation = _broadcast_modulation(x, modulation_or_weight)
        scale, shift, gate = modulation.chunk(3, dim=-1)
        y = normed * (1.0 + scale.to(normed.dtype)) + shift.to(normed.dtype)
        return y.to(x.dtype), gate.to(x.dtype), rstd
    y = normed * (1.0 + modulation_or_weight.to(normed.dtype))
    return y.to(x.dtype), None, rstd

def resgate_adarms_forward(x: torch.Tensor, h: torch.Tensor, gate_in: torch.Tensor | None, modulation_or_weight: torch.Tensor, eps: float = 1e-6, adaptive: bool = True):
    residual = x + h if gate_in is None else x + h * gate_in
    y, gate, rstd = adarms_forward(residual, modulation_or_weight, eps, adaptive)
    return residual, y, gate, rstd

class FlashRTAdaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, cond_dim: int | None = None) -> None:
        super().__init__()
        self.dim = int(dim); self.eps = float(eps); self.cond_dim = cond_dim
        if cond_dim is None:
            self.weight = nn.Parameter(torch.zeros(dim)); self.dense = None
        else:
            self.weight = None; self.dense = nn.Linear(cond_dim, dim * 3, bias=True); nn.init.zeros_(self.dense.weight)
    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None):
        if self.dense is None:
            return adarms_forward(x, self.weight, self.eps, adaptive=False)[:2]
        if cond is None: raise ValueError("cond is required for adaptive FlashRTAdaRMSNorm")
        return adarms_forward(x, self.dense(cond), self.eps, adaptive=True)[:2]

def backend_marker(x: torch.Tensor) -> torch.Tensor:
    return ops._flashrt_training_package_marker(x)
__all__ = ["adarms_forward", "resgate_adarms_forward", "FlashRTAdaRMSNorm", "backend_marker"]
