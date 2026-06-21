#!/usr/bin/env python3
"""Correctness tests for bf16-linear-gemv."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "bf16-linear-gemv"
REGISTRATION_INCLUDE = ROOT.parent / "kernels" / "kernel-builder" / "src" / "pyproject" / "templates" / "torch"


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self.ops = getattr(torch.ops, namespace)

    def bf16_decode_gemv_bf16(self, x, w, alpha=1.0, variant=0, out=None):
        if out is None:
            out = torch.empty((w.shape[0],), device=x.device, dtype=torch.bfloat16)
        self.ops.bf16_decode_gemv_bf16(x, w, float(alpha), int(variant), out)
        return out

    def bf16_decode_gemv_unrolled_bf16(self, x, w, out=None):
        if out is None:
            out = torch.empty((w.shape[0],), device=x.device, dtype=torch.bfloat16)
        self.ops.bf16_decode_gemv_unrolled_bf16(x, w, out)
        return out


def _arch_list() -> str:
    major, minor = torch.cuda.get_device_capability(0)
    return "12.0a" if major >= 12 else f"{major}.{minor}"


def load_source_ops() -> SourceOps:
    from torch.utils.cpp_extension import load

    if not REGISTRATION_INCLUDE.is_dir():
        raise RuntimeError(f"missing registration include: {REGISTRATION_INCLUDE}")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _arch_list())
    namespace = "bf16_linear_gemv_source_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "gemm" / "bf16_gemv_m1_sm120.cu"),
            str(PACKAGE / "csrc" / "kernels" / "nexn2_bf16_gemv.cu"),
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
        return importlib.import_module("bf16_linear_gemv")
    finally:
        if artifact:
            sys.path.remove(artifact)


def assert_close(name: str, got: torch.Tensor, ref: torch.Tensor) -> None:
    diff = (got.float() - ref.float()).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    cos = float(torch.nn.functional.cosine_similarity(got.float(), ref.float(), dim=0).item())
    if max_abs > 0.125 or mean_abs > 0.006 or cos < 0.999:
        raise AssertionError(f"{name}: max_abs={max_abs:.8f} mean_abs={mean_abs:.8f} cos={cos:.8f}")


def run(ops, mode: str) -> int:
    shapes = [(1024, 1024), (4096, 1024)] if mode == "smoke" else [
        (1024, 1024), (2048, 4096), (4096, 4096), (8192, 4096), (12288, 5120)
    ]
    count = 0
    for n, k in shapes:
        gen = torch.Generator(device="cuda").manual_seed(1000 + n + k)
        x = (torch.randn((k,), device="cuda", generator=gen) * 0.05).to(torch.bfloat16).contiguous()
        w = (torch.randn((n, k), device="cuda", generator=gen) * 0.05).to(torch.bfloat16).contiguous()
        ref = (x.float() @ w.float().t()).to(torch.bfloat16)
        for variant in (0, 4, 8):
            got = ops.bf16_decode_gemv_bf16(x, w, variant=variant)
            torch.cuda.synchronize()
            assert_close(f"bf16_decode_gemv_bf16 n={n} k={k} v={variant}", got, ref)
            count += 1
        got = ops.bf16_decode_gemv_unrolled_bf16(x, w)
        torch.cuda.synchronize()
        assert_close(f"bf16_decode_gemv_unrolled_bf16 n={n} k={k}", got, ref)
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    count = run(ops, args.mode)
    print(f"bf16-linear-gemv {args.backend} {args.mode}: passed {count}/{count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
