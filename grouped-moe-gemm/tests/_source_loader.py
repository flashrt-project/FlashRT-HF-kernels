from __future__ import annotations
import os
from pathlib import Path
import torch

PACKAGE = Path(__file__).resolve().parents[1]


class SourceOps:
    def __init__(self, ns, functional):
        self.ops = getattr(torch.ops, ns)
        self.functional = functional

    def grouped_nvfp4_gemm_bf16(
        self,
        a,
        w,
        sa,
        sw,
        alpha,
        experts,
        *,
        tile_rows,
        input_scale_stride=0,
        weight_stride=None,
        weight_scale_stride=None,
        out=None,
    ):
        weight_stride = w[0].numel() if weight_stride is None else weight_stride
        weight_scale_stride = (
            sw[0].numel() if weight_scale_stride is None else weight_scale_stride
        )
        if out is None:
            return self.functional(
                a,
                w,
                sa,
                sw,
                alpha,
                experts,
                int(tile_rows),
                int(input_scale_stride),
                int(weight_stride),
                int(weight_scale_stride),
            )
        self.ops.grouped_nvfp4_gemm_bf16_out(
            a,
            w,
            sa,
            sw,
            alpha,
            experts,
            tile_rows,
            input_scale_stride,
            weight_stride,
            weight_scale_stride,
            out,
        )
        return out


def load_source_ops(registration_include=None):
    from torch.utils.cpp_extension import load

    include = registration_include or os.environ.get(
        "KERNEL_BUILDER_REGISTRATION_INCLUDE"
    )
    if not include:
        include = str(
            PACKAGE.parent.parent
            / "kernels/kernel-builder/src/pyproject/templates/torch"
        )
    cutlass = os.environ.get(
        "CUTLASS_INCLUDE",
        "/home/heima/suliang/PI/official/FlashRT/third_party/cutlass/include",
    )
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0a"
    ns = "grouped_moe_gemm_source_test"
    load(
        name=ns,
        sources=[
            str(PACKAGE / "torch-ext/torch_binding.cpp"),
            *map(str, sorted((PACKAGE / "csrc").glob("*.cu"))),
        ],
        extra_include_paths=[str(PACKAGE / "csrc"), include, cutlass],
        extra_cflags=["-O3", "-DCUDA_KERNEL"],
        extra_cuda_cflags=["-O3", "-DCUDA_KERNEL"],
        is_python_module=False,
        verbose=False,
    )
    raw = getattr(torch.ops, ns).grouped_nvfp4_gemm_bf16_out
    wns = "grouped_moe_gemm_source_wrapper"

    @torch.library.custom_op(f"{wns}::run", mutates_args=(), device_types="cuda")
    def functional(
        a: torch.Tensor,
        w: torch.Tensor,
        sa: torch.Tensor,
        sw: torch.Tensor,
        alpha: torch.Tensor,
        experts: torch.Tensor,
        tile_rows: int,
        input_scale_stride: int,
        weight_stride: int,
        weight_scale_stride: int,
    ) -> torch.Tensor:
        out = torch.empty(
            (a.shape[0], w.shape[1]), device=a.device, dtype=torch.bfloat16
        )
        raw(
            a,
            w,
            sa,
            sw,
            alpha,
            experts,
            tile_rows,
            input_scale_stride,
            weight_stride,
            weight_scale_stride,
            out,
        )
        return out

    @torch.library.register_fake(f"{wns}::run")
    def fake(
        a,
        w,
        sa,
        sw,
        alpha,
        experts,
        tile_rows,
        input_scale_stride,
        weight_stride,
        weight_scale_stride,
    ):
        return torch.empty(
            (a.shape[0], w.shape[1]), device=a.device, dtype=torch.bfloat16
        )

    return SourceOps(ns, functional)
