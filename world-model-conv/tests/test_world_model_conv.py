#!/usr/bin/env python3
"""Correctness tests for world-model-conv."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import importlib
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "world-model-conv"
REGISTRATION_INCLUDE = (
    ROOT.parent
    / "kernels"
    / "kernel-builder"
    / "src"
    / "pyproject"
    / "templates"
    / "torch"
)


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self._ops = getattr(torch.ops, namespace)

    def fp8_conv3d_v18_ncdhw_res_bf16out(self, cache_x, new_x, weight, bias, residual, alpha=1.0):
        out = torch.empty_like(residual)
        self._ops.fp8_conv3d_v18_ncdhw_res_bf16out(cache_x, new_x, weight, bias, residual, float(alpha), out)
        return out


def _preload_cublaslt() -> None:
    for parent in Path(torch.__file__).resolve().parents:
        candidate = parent / "nvidia" / "cublas" / "lib" / "libcublasLt.so.12"
        if candidate.exists():
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
            return
    library = ctypes.util.find_library("cublasLt")
    if library:
        ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)


def _current_arch_list() -> str:
    major, minor = torch.cuda.get_device_capability(0)
    return f"{major}.{minor}"


def load_source_ops() -> SourceOps:
    from torch.utils.cpp_extension import load

    if not REGISTRATION_INCLUDE.is_dir():
        raise RuntimeError(f"missing kernel-builder registration include: {REGISTRATION_INCLUDE}")
    _preload_cublaslt()
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0a"
    namespace = "world_model_conv_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "fp8_conv3d_sm120_v18.cu"),
        ],
        extra_include_paths=[str(PACKAGE / "csrc"), str(REGISTRATION_INCLUDE)],
        extra_cflags=["-O3", "-DCUDA_KERNEL"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-DCUDA_KERNEL"],
        is_python_module=False,
        verbose=False,
    )
    return SourceOps(namespace)


def load_installed_ops(artifact: str | None):
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return importlib.import_module("world_model_conv")
    finally:
        if artifact:
            sys.path.remove(artifact)


def ref_conv(cache_x, new_x, weight, bias, residual, alpha: float) -> torch.Tensor:
    x = torch.cat([cache_x, new_x], dim=1).to(torch.float32)
    # NDHWC -> NCDHW
    x_ncdhw = x.permute(0, 4, 1, 2, 3).contiguous()
    # (Co,3,3,3,Ci) -> (Co,Ci,3,3,3)
    w = weight.to(torch.float32).permute(0, 4, 1, 2, 3).contiguous()
    y = F.conv3d(x_ncdhw, w, bias=None, stride=1, padding=(0, 1, 1))
    y = y * float(alpha) + bias.float().view(1, -1, 1, 1, 1)
    y_bf16 = y.to(torch.bfloat16)
    return (y_bf16.float() + residual.float()).to(torch.bfloat16)


def assert_close(name: str, got: torch.Tensor, ref: torch.Tensor, max_atol: float, mean_atol: float, min_cos: float) -> None:
    diff = (got.float() - ref.float()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    cos = torch.nn.functional.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0).item()
    if max_err > max_atol or mean_err > mean_atol or cos < min_cos:
        raise AssertionError(f"{name}: max_err={max_err:.8f}, mean_err={mean_err:.8f}, cos={cos:.8f}")


def run_tests(ops) -> int:
    count = 0
    shapes = [
        (1, 2, 1, 8, 8, 32, 16),
        (1, 2, 4, 16, 16, 32, 32),
        (2, 2, 4, 16, 24, 64, 32),
    ]
    for n, tc, tn, h, w, ci, co in shapes:
        cache = (torch.randn((n, tc, h, w, ci), device="cuda") * 0.1).to(torch.float8_e4m3fn)
        new = (torch.randn((n, tn, h, w, ci), device="cuda") * 0.1).to(torch.float8_e4m3fn)
        weight = (torch.randn((co, 3, 3, 3, ci), device="cuda") * 0.1).to(torch.float8_e4m3fn)
        bias = (torch.randn((co,), device="cuda") * 0.01).to(torch.bfloat16)
        residual = (torch.randn((n, co, tn, h, w), device="cuda") * 0.05).to(torch.bfloat16)
        alpha = 0.75
        got = ops.fp8_conv3d_v18_ncdhw_res_bf16out(cache, new, weight, bias, residual, alpha)
        ref = ref_conv(cache, new, weight, bias, residual, alpha)
        assert_close(f"fp8_conv3d shape={(n,tc,tn,h,w,ci,co)}", got, ref, 0.125, 0.01, 0.999)
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    major, _ = torch.cuda.get_device_capability(0)
    if major < 12:
        raise RuntimeError("world-model-conv source validation requires Blackwell SM120+")
    torch.manual_seed(0)
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    total = run_tests(ops)
    torch.cuda.synchronize()
    print(f"world-model-conv correctness passed: {total} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
