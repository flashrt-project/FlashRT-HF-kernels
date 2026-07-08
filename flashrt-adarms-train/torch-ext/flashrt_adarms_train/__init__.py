"""FlashRT AdaRMS training kernels: fused AdaRMS / gated-residual+AdaRMS
forward and backward.

Operator (fp32 internal math):
    rstd = rsqrt(mean(x^2, -1) + eps); xhat = x * rstd
    adaptive:     y = xhat * (1 + scale) + shift
    non-adaptive: y = xhat * (1 + weight)
The resgate variant computes r = x + h * gate_in first (r is a real output)
and normalizes r.

The public API dispatches to the CUDA kernels when supported (CUDA, bf16 or
fp32, H a multiple of 8/4 up to the register-cached limit) and to the pure
PyTorch reference otherwise. The modulation projection (``dense(cond)``) and
the gate chunk stay in PyTorch, so their autograd is native.
"""

from __future__ import annotations

import torch
from torch import nn

try:
    from ._ops import ops

    _HAS_OPS = hasattr(ops, "adarms_fwd")
except Exception:  # source-tree tests before kernel-builder creates _ops.py
    ops = None
    _HAS_OPS = False


def _use_ops(namespace_ops) -> None:
    """Install a manually built extension (dev/testing path)."""
    global ops, _HAS_OPS
    ops = namespace_ops
    _HAS_OPS = hasattr(ops, "adarms_fwd")
    _register_fakes()


# ---------------------------------------------------------------------------
# pure-PyTorch reference (also the CPU / unsupported-shape fallback)
# ---------------------------------------------------------------------------


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


def reference_adarms(
    x: torch.Tensor,
    modulation_or_weight: torch.Tensor,
    eps: float = 1e-6,
    adaptive: bool = True,
):
    normed, rstd = _rms_norm(x, eps)
    if adaptive:
        modulation = _broadcast_modulation(x, modulation_or_weight)
        scale, shift, gate = modulation.chunk(3, dim=-1)
        y = normed * (1.0 + scale.to(normed.dtype)) + shift.to(normed.dtype)
        return y.to(x.dtype), gate.to(x.dtype), rstd
    y = normed * (1.0 + modulation_or_weight.to(normed.dtype))
    return y.to(x.dtype), None, rstd


def reference_resgate_adarms(
    x: torch.Tensor,
    h: torch.Tensor,
    gate_in: torch.Tensor | None,
    modulation_or_weight: torch.Tensor,
    eps: float = 1e-6,
    adaptive: bool = True,
):
    residual = x + h if gate_in is None else x + h * gate_in
    y, gate, rstd = reference_adarms(residual, modulation_or_weight, eps, adaptive)
    return residual, y, gate, rstd


# ---------------------------------------------------------------------------
# CUDA autograd path
# ---------------------------------------------------------------------------


def _kernel_supported(x: torch.Tensor) -> bool:
    if not (_HAS_OPS and x.is_cuda and x.dim() == 3):
        return False
    if x.dtype not in (torch.bfloat16, torch.float32):
        return False
    lanes = 8 if x.dtype == torch.bfloat16 else 4
    return x.shape[-1] % lanes == 0 and x.shape[-1] <= 256 * lanes * 4


def _reduce_broadcast(grad: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    if like.shape[1] == 1 and grad.shape[1] != 1:
        return grad.sum(dim=1, keepdim=True)
    return grad


def _register_fakes() -> None:
    """Fake (meta) kernels so torch.compile can trace through the ops."""
    if not _HAS_OPS:
        return
    try:
        namespace = ops.adarms_fwd.default.name().split("::")[0]

        def _rstd_like(x):
            return x.new_empty((x.shape[0] * x.shape[1],), dtype=torch.float32)

        @torch.library.register_fake(f"{namespace}::adarms_fwd")
        def _(x, scale, shift, weight, eps):
            return torch.empty_like(x), _rstd_like(x)

        @torch.library.register_fake(f"{namespace}::adarms_bwd")
        def _(dy, x, scale, weight, rstd):
            dmod = (
                torch.empty_like(x)
                if scale is not None
                else x.new_empty((x.shape[-1],), dtype=torch.float32)
            )
            return torch.empty_like(x), dmod

        @torch.library.register_fake(f"{namespace}::resgate_adarms_fwd")
        def _(x, h, gate, scale, shift, weight, eps):
            return torch.empty_like(x), torch.empty_like(x), _rstd_like(x)

        @torch.library.register_fake(f"{namespace}::resgate_adarms_bwd")
        def _(dy, dyr, r, h, gate, scale, weight, rstd):
            dg = torch.empty_like(r) if gate is not None else r.new_empty((0,))
            dmod = (
                torch.empty_like(r)
                if scale is not None
                else r.new_empty((r.shape[-1],), dtype=torch.float32)
            )
            return torch.empty_like(r), torch.empty_like(r), dg, dmod
    except Exception:
        pass  # fakes are a compile nicety, never a functional requirement


_register_fakes()


class _AdaRMSFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, shift, weight, eps):
        y, rstd = ops.adarms_fwd(x, scale, shift, weight, float(eps))
        ctx.save_for_backward(x, scale, weight, rstd)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, scale, weight, rstd = ctx.saved_tensors
        dy = dy.contiguous()
        dx, dmod = ops.adarms_bwd(dy, x, scale, weight, rstd)
        if scale is not None:
            dscale = _reduce_broadcast(dmod, scale)
            dshift = _reduce_broadcast(dy, scale)
            return dx, dscale, dshift, None, None
        return dx, None, None, dmod.to(weight.dtype), None


class _ResGateAdaRMSFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, h, gate_in, scale, shift, weight, eps):
        r, y, rstd = ops.resgate_adarms_fwd(
            x, h, gate_in, scale, shift, weight, float(eps)
        )
        ctx.save_for_backward(h, gate_in, scale, weight, r, rstd)
        return r, y

    @staticmethod
    def backward(ctx, dr_out, dy):
        h, gate_in, scale, weight, r, rstd = ctx.saved_tensors
        dy = dy.contiguous()
        dyr = dr_out.contiguous() if dr_out is not None else None
        dr, dh, dg, dmod = ops.resgate_adarms_bwd(
            dy, dyr, r, h, gate_in, scale, weight, rstd
        )
        dg_out = dg if gate_in is not None else None
        if scale is not None:
            dscale = _reduce_broadcast(dmod, scale)
            dshift = _reduce_broadcast(dy, scale)
            return dr, dh, dg_out, dscale, dshift, None, None
        return dr, dh, dg_out, None, None, dmod.to(weight.dtype), None


def _split_modulation(x: torch.Tensor, modulation: torch.Tensor):
    modulation = _broadcast_modulation(x, modulation)
    scale, shift, gate = modulation.chunk(3, dim=-1)
    return scale, shift, gate


def adarms(
    x: torch.Tensor,
    modulation_or_weight: torch.Tensor,
    eps: float = 1e-6,
    adaptive: bool = True,
):
    """AdaRMS with fused CUDA fwd+bwd. Returns (y, gate_or_None)."""
    if not _kernel_supported(x):
        y, gate, _ = reference_adarms(x, modulation_or_weight, eps, adaptive)
        return y, gate
    if adaptive:
        scale, shift, gate = _split_modulation(x, modulation_or_weight)
        y = _AdaRMSFn.apply(x.contiguous(), scale, shift, None, eps)
        return y, gate
    y = _AdaRMSFn.apply(x.contiguous(), None, None, modulation_or_weight, eps)
    return y, None


def resgate_adarms(
    x: torch.Tensor,
    h: torch.Tensor,
    gate_in: torch.Tensor | None,
    modulation_or_weight: torch.Tensor,
    eps: float = 1e-6,
    adaptive: bool = True,
):
    """r = x + h * gate_in, then AdaRMS(r). Returns (r, y, gate_or_None)."""
    if not _kernel_supported(x):
        r, y, gate, _ = reference_resgate_adarms(
            x, h, gate_in, modulation_or_weight, eps, adaptive
        )
        return r, y, gate
    gate_in_c = None
    if gate_in is not None:
        gate_in_c = (
            gate_in.expand_as(x).contiguous()
            if gate_in.shape != x.shape
            else gate_in.contiguous()
        )
    if adaptive:
        scale, shift, gate = _split_modulation(x, modulation_or_weight)
        r, y = _ResGateAdaRMSFn.apply(
            x.contiguous(), h.contiguous(), gate_in_c, scale, shift, None, eps
        )
        return r, y, gate
    r, y = _ResGateAdaRMSFn.apply(
        x.contiguous(), h.contiguous(), gate_in_c, None, None, modulation_or_weight, eps
    )
    return r, y, None


class FlashRTAdaRMSNorm(nn.Module):
    """Drop-in for pi-style adaptive RMSNorm: forward(x, cond) -> (y, gate)."""

    def __init__(self, dim: int, eps: float = 1e-6, cond_dim: int | None = None) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.cond_dim = cond_dim
        if cond_dim is None:
            self.weight = nn.Parameter(torch.zeros(dim))
            self.dense = None
        else:
            self.weight = None
            self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
            nn.init.zeros_(self.dense.weight)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None):
        if self.dense is None:
            return adarms(x, self.weight, self.eps, adaptive=False)
        if cond is None:
            raise ValueError("cond is required for adaptive FlashRTAdaRMSNorm")
        return adarms(x, self.dense(cond), self.eps, adaptive=True)


def backend_marker(x: torch.Tensor) -> torch.Tensor:
    if ops is None:
        return x
    return ops._flashrt_training_package_marker(x)


__all__ = [
    "adarms",
    "resgate_adarms",
    "reference_adarms",
    "reference_resgate_adarms",
    "FlashRTAdaRMSNorm",
    "backend_marker",
]
