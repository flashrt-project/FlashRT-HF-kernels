#!/usr/bin/env python3
"""Minimal flashrt-spatiotemporal-layout example."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import importlib
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "flashrt-spatiotemporal-layout"
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

    def ncdhw_to_blc_bf16(self, x, out=None):
        if out is None:
            out = torch.empty((x.shape[0], x.shape[2] * x.shape[3] * x.shape[4], x.shape[1]), device=x.device, dtype=x.dtype)
        self._ops.ncdhw_to_blc_bf16(x, out)
        return out

    def time_unshuffle2_bf16(self, x, out=None):
        if out is None:
            out = torch.empty((x.shape[0], x.shape[1] // 2, 2 * x.shape[2], x.shape[3], x.shape[4]), device=x.device, dtype=x.dtype)
        self._ops.time_unshuffle2_bf16(x, out)
        return out

    def add_bias_ncdhw_bf16(self, x, bias):
        self._ops.add_bias_ncdhw_bf16(x, bias)
        return x

    def update_cache2_ncdhw_bf16(self, cur, prev, out=None):
        if out is None:
            out = torch.empty((cur.shape[0], cur.shape[1], 2, cur.shape[3], cur.shape[4]), device=cur.device, dtype=cur.dtype)
        self._ops.update_cache2_ncdhw_bf16(cur, prev, out)
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


def load_source_ops():
    from torch.utils.cpp_extension import load

    if not REGISTRATION_INCLUDE.is_dir():
        raise RuntimeError(f"missing kernel-builder registration include: {REGISTRATION_INCLUDE}")
    _preload_cublaslt()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _current_arch_list())
    namespace = "flashrt_spatiotemporal_layout_example"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "spatiotemporal_layout.cu"),
        ],
        extra_include_paths=[str(PACKAGE / "csrc"), str(REGISTRATION_INCLUDE)],
        extra_cflags=["-O3", "-DCUDA_KERNEL"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-DCUDA_KERNEL"],
        verbose=False,
    )
    return SourceOps(namespace)


def load_hub_ops(version: int):
    from kernels import get_kernel

    return get_kernel(
        "flashrt/flashrt-spatiotemporal-layout",
        version=version,
        trust_remote_code=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("source", "hub"), default="source")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    ops = load_source_ops() if args.backend == "source" else load_hub_ops(args.version)

    ncdhw_to_blc = ops.ncdhw_to_blc_bf16
    time_unshuffle2 = ops.time_unshuffle2_bf16
    if args.compile:
        ncdhw_to_blc = torch.compile(ncdhw_to_blc, fullgraph=True)
        time_unshuffle2 = torch.compile(time_unshuffle2, fullgraph=True)

    x = torch.randn((1, 64, 4, 32, 32), device="cuda", dtype=torch.bfloat16)
    x2 = torch.randn((1, 128, 4, 32, 32), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((64,), device="cuda", dtype=torch.bfloat16)
    prev = torch.randn((1, 64, 2, 32, 32), device="cuda", dtype=torch.bfloat16)

    tokens = ncdhw_to_blc(x)
    expanded = time_unshuffle2(x2)
    ops.add_bias_ncdhw_bf16(x, bias)
    cache = ops.update_cache2_ncdhw_bf16(x[:, :, -1:, :, :].contiguous(), prev)
    torch.cuda.synchronize()

    print("tokens", tuple(tokens.shape), tokens.dtype)
    print("expanded", tuple(expanded.shape), expanded.dtype)
    print("cache", tuple(cache.shape), cache.dtype)


if __name__ == "__main__":
    main()
