"""Build and load the package directly from source for local validation."""

from __future__ import annotations

import os
from pathlib import Path

import torch

PACKAGE = Path(__file__).resolve().parents[1]


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self.ops = getattr(torch.ops, namespace)

    def cholesky_small_fp32(
        self,
        input: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if out is None:
            out = torch.empty_like(input)
        self.ops.cholesky_small_fp32_out(input, out)
        return out


def _arch_list() -> str:
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return f"{major}.{minor}"


def load_source_ops(registration_include: str | None) -> SourceOps:
    from torch.utils.cpp_extension import load

    include = registration_include or os.environ.get(
        "KERNEL_BUILDER_REGISTRATION_INCLUDE"
    )
    if not include:
        candidate = (
            PACKAGE.parent.parent
            / "kernels"
            / "kernel-builder"
            / "src"
            / "pyproject"
            / "templates"
            / "torch"
        )
        if candidate.is_dir():
            include = str(candidate)
    if not include or not (Path(include) / "registration.h").is_file():
        raise RuntimeError(
            "registration.h not found; pass --registration-include or set "
            "KERNEL_BUILDER_REGISTRATION_INCLUDE"
        )

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _arch_list())
    namespace = "small_matrix_cholesky_source_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "cholesky_small_fp32.cu"),
        ],
        extra_include_paths=[str(PACKAGE / "csrc"), include],
        extra_cflags=["-O3", "-DCUDA_KERNEL"],
        extra_cuda_cflags=["-O3", "-DCUDA_KERNEL"],
        is_python_module=False,
        verbose=False,
    )
    return SourceOps(namespace)
