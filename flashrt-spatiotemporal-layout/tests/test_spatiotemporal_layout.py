#!/usr/bin/env python3
"""Correctness tests for flashrt-spatiotemporal-layout."""

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

    def patch_im2col_bf16(self, x, out=None):
        if out is None:
            out = torch.empty((x.shape[0] * 256, 588), device=x.device, dtype=x.dtype)
        self._ops.patch_im2col_bf16(x, out)
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


def load_source_ops() -> SourceOps:
    from torch.utils.cpp_extension import load

    if not REGISTRATION_INCLUDE.is_dir():
        raise RuntimeError(f"missing kernel-builder registration include: {REGISTRATION_INCLUDE}")
    _preload_cublaslt()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _current_arch_list())
    namespace = "flashrt_spatiotemporal_layout_test"
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


def load_installed_ops(artifact: str | None):
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return importlib.import_module("flashrt_spatiotemporal_layout")
    finally:
        if artifact:
            sys.path.remove(artifact)


def assert_exact(name: str, got: torch.Tensor, expected: torch.Tensor) -> None:
    diff = (got.float() - expected.float()).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    if max_abs != 0.0:
        raise AssertionError(f"{name} failed: max_abs={max_abs}")
    print(f"PASS {name}: max_abs=0")


def ref_cache2(cur: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
    out = torch.empty((cur.shape[0], cur.shape[1], 2, cur.shape[3], cur.shape[4]), device=cur.device, dtype=cur.dtype)
    if cur.shape[2] >= 2:
        out.copy_(cur[:, :, -2:, :, :])
    else:
        out[:, :, 0].copy_(prev[:, :, 1])
        out[:, :, 1].copy_(cur[:, :, 0])
    return out


def ref_patch_im2col(x: torch.Tensor) -> torch.Tensor:
    return (
        x.reshape(x.shape[0], 16, 14, 16, 14, 3)
        .permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(x.shape[0] * 256, 588)
    )


def run_shape(ops, label: str, shape: tuple[int, int, int, int, int]) -> None:
    b, c, t, h, w = shape
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    got_blc = ops.ncdhw_to_blc_bf16(x)
    exp_blc = x.permute(0, 2, 3, 4, 1).contiguous().view(b, t * h * w, c)
    assert_exact(f"{label}/ncdhw_to_blc", got_blc, exp_blc)

    x2 = torch.randn((b, 2 * c, t, h, w), device="cuda", dtype=torch.bfloat16)
    got_unshuffle = ops.time_unshuffle2_bf16(x2)
    exp_unshuffle = torch.empty((b, c, 2 * t, h, w), device="cuda", dtype=torch.bfloat16)
    exp_unshuffle[:, :, 0::2] = x2[:, :c]
    exp_unshuffle[:, :, 1::2] = x2[:, c:]
    assert_exact(f"{label}/time_unshuffle2", got_unshuffle, exp_unshuffle)

    bias = torch.randn((c,), device="cuda", dtype=torch.bfloat16)
    x_bias = x.clone()
    got_bias = ops.add_bias_ncdhw_bf16(x_bias, bias)
    exp_bias = (x.float() + bias.float().view(1, c, 1, 1, 1)).to(torch.bfloat16)
    assert_exact(f"{label}/add_bias_ncdhw", got_bias, exp_bias)

    prev = torch.randn((b, c, 2, h, w), device="cuda", dtype=torch.bfloat16)
    got_cache = ops.update_cache2_ncdhw_bf16(x, prev)
    assert_exact(f"{label}/update_cache2", got_cache, ref_cache2(x, prev))

    x_t1 = torch.randn((b, c, 1, h, w), device="cuda", dtype=torch.bfloat16)
    got_cache_t1 = ops.update_cache2_ncdhw_bf16(x_t1, prev)
    assert_exact(f"{label}/update_cache2_t1", got_cache_t1, ref_cache2(x_t1, prev))


def run_patch_shape(ops, num_views: int) -> None:
    x = torch.randn((num_views, 224, 224, 3), device="cuda", dtype=torch.bfloat16)
    got = ops.patch_im2col_bf16(x)
    assert_exact(f"patch_nv{num_views}/patch_im2col", got, ref_patch_im2col(x))


def run(args) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(59)
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    shapes = {
        "small": (1, 8, 4, 8, 8),
        "latent_16": (1, 16, 8, 32, 32),
        "latent_64": (1, 64, 4, 32, 32),
    }
    if args.mode == "smoke":
        shapes = {"small": shapes["small"]}
    for label, shape in shapes.items():
        run_shape(ops, label, shape)
    run_patch_shape(ops, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
