from __future__ import annotations
import os
from pathlib import Path
import torch

PACKAGE = Path(__file__).resolve().parents[1]


class SourceOps:
    def __init__(self, ns, gated_functional, residual_functional):
        self.ops = getattr(torch.ops, ns)
        self.gated_functional = gated_functional
        self.residual_functional = residual_functional

    def gated(self, x, uw, ub, dinv, dw, db, g, r, ua, da, hs, out, scr):
        self.ops.fp8_gelu_ffn_gated_residual_bf16_out(
            x, uw, ub, dinv, dw, db, g, r, ua, da, hs, out, scr
        )
        return out

    def residual(
        self, x, uinv, uw, ub, dinv, dw, db, r, ua, da, us, ds, split, out, xs, hs, b
    ):
        self.ops.fp8_gelu_ffn_residual_bf16_out(
            x, uinv, uw, ub, dinv, dw, db, r, ua, da, us, ds, split, out, xs, hs, b
        )
        return out


def load_source_ops(registration_include=None):
    from torch.utils.cpp_extension import load

    inc = (
        registration_include
        or os.environ.get("KERNEL_BUILDER_REGISTRATION_INCLUDE")
        or str(
            PACKAGE.parent.parent
            / "kernels/kernel-builder/src/pyproject/templates/torch"
        )
    )
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0a"
    ns = "smallm_ffn_megakernels_blackwell_source_test"
    load(
        name=ns,
        sources=[
            str(PACKAGE / "torch-ext/torch_binding.cpp"),
            *map(str, sorted((PACKAGE / "csrc").glob("*.cu"))),
        ],
        extra_include_paths=[str(PACKAGE / "csrc"), inc],
        extra_cflags=["-O3", "-DCUDA_KERNEL"],
        extra_cuda_cflags=["-O3", "-DCUDA_KERNEL"],
        is_python_module=False,
        verbose=False,
    )
    raw = getattr(torch.ops, ns)
    wns = "smallm_ffn_megakernels_source_wrapper"

    @torch.library.custom_op(f"{wns}::gated", mutates_args=(), device_types="cuda")
    def gated(
        x: torch.Tensor,
        uw: torch.Tensor,
        ub: torch.Tensor,
        dinv: torch.Tensor,
        dw: torch.Tensor,
        db: torch.Tensor,
        g: torch.Tensor,
        r: torch.Tensor,
        ua: float,
        da: float,
        scale: float,
    ) -> torch.Tensor:
        out = torch.empty_like(r)
        scr = torch.empty(
            (x.shape[0], 4096), device=x.device, dtype=torch.float8_e4m3fn
        )
        raw.fp8_gelu_ffn_gated_residual_bf16_out(
            x, uw, ub, dinv, dw, db, g, r, ua, da, scale, out, scr
        )
        return out

    @torch.library.register_fake(f"{wns}::gated")
    def gated_fake(x, uw, ub, dinv, dw, db, g, r, ua, da, scale):
        return torch.empty_like(r)

    @torch.library.custom_op(f"{wns}::residual", mutates_args=(), device_types="cuda")
    def residual(
        x: torch.Tensor,
        uinv: torch.Tensor,
        uw: torch.Tensor,
        ub: torch.Tensor,
        dinv: torch.Tensor,
        dw: torch.Tensor,
        db: torch.Tensor,
        r: torch.Tensor,
        ua: float,
        da: float,
        us: float,
        ds: float,
        split: bool,
    ) -> torch.Tensor:
        out = torch.empty_like(r)
        xs = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        hs = torch.empty((x.shape[0], 2048), device=x.device, dtype=torch.float8_e4m3fn)
        bar = torch.zeros(2, device=x.device, dtype=torch.uint32)
        raw.fp8_gelu_ffn_residual_bf16_out(
            x, uinv, uw, ub, dinv, dw, db, r, ua, da, us, ds, split, out, xs, hs, bar
        )
        return out

    @torch.library.register_fake(f"{wns}::residual")
    def residual_fake(x, uinv, uw, ub, dinv, dw, db, r, ua, da, us, ds, split):
        return torch.empty_like(r)

    return SourceOps(ns, gated, residual)
