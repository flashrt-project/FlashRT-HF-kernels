#!/usr/bin/env python3
"""Correctness tests for flashrt-gemm-epilogues BF16 GEMM bias/GELU ops."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "flashrt-gemm-epilogues"
REGISTRATION_INCLUDE = (
    ROOT.parent
    / "kernels"
    / "kernel-builder"
    / "src"
    / "pyproject"
    / "templates"
    / "torch"
)

_SOURCE_LIST = [
    "torch-ext/torch_binding.cpp",
    "csrc/bf16_gemm_bias_gelu.cu",
    "csrc/bias_gelu_quantize_fp8.cu",
    "csrc/channel_scale_quantize_fp8.cu",
]


def _arch_list() -> str:
    major, minor = torch.cuda.get_device_capability(0)
    if major >= 12:
        return "12.0a"
    if (major, minor) == (11, 0):
        return "11.0a"
    return f"{major}.{minor}"


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self._ops = getattr(torch.ops, namespace)

    def gemm_bias(self, a, b, bias, *, out=None):
        if out is None:
            out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.bfloat16)
        self._ops.bf16_gemm_bias(a, b, bias, out)
        return out

    def gemm_bias_gelu(self, a, b, bias, *, out=None):
        if out is None:
            out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.bfloat16)
        self._ops.bf16_gemm_bias_gelu(a, b, bias, out)
        return out


class InstalledOps:
    def __init__(self, module) -> None:
        self._module = module

    def gemm_bias(self, a, b, bias, *, out=None):
        return self._module.bf16_gemm_bias(a, b, bias, out=out)

    def gemm_bias_gelu(self, a, b, bias, *, out=None):
        return self._module.bf16_gemm_bias_gelu(a, b, bias, out=out)


def _preload_cublaslt() -> None:
    import ctypes
    import ctypes.util

    for parent in Path(torch.__file__).resolve().parents:
        candidate = parent / "nvidia" / "cublas" / "lib" / "libcublasLt.so.12"
        if candidate.exists():
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
            return
    library = ctypes.util.find_library("cublasLt")
    if library:
        ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)


def load_source_ops() -> SourceOps:
    from torch.utils.cpp_extension import load

    if not REGISTRATION_INCLUDE.is_dir():
        raise RuntimeError(f"missing kernel-builder registration include: {REGISTRATION_INCLUDE}")
    _preload_cublaslt()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _arch_list())
    namespace = "flashrt_gemm_epilogues_test"
    load(
        name=namespace,
        sources=[str(PACKAGE / s) for s in _SOURCE_LIST],
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
        return InstalledOps(importlib.import_module("flashrt_gemm_epilogues"))
    finally:
        if artifact:
            sys.path.remove(artifact)


def _check_gemm_bias(ops, m: int, n: int, k: int) -> None:
    torch.manual_seed(2)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16).contiguous()
    b = torch.randn((k, n), device="cuda", dtype=torch.bfloat16).contiguous()
    bias = torch.randn((n,), device="cuda", dtype=torch.bfloat16).contiguous()
    out = ops.gemm_bias(a, b, bias)
    expected = ((a @ b) + bias).to(torch.bfloat16)
    torch.testing.assert_close(out.float(), expected.float(), rtol=3e-2, atol=1.25e-1)
    print(f"PASS bf16_gemm_bias ({m},{n},{k})")


def _check_gemm_bias_gelu(ops, m: int, n: int, k: int) -> None:
    torch.manual_seed(3)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16).contiguous()
    b = torch.randn((k, n), device="cuda", dtype=torch.bfloat16).contiguous()
    bias = torch.randn((n,), device="cuda", dtype=torch.bfloat16).contiguous()
    out = ops.gemm_bias_gelu(a, b, bias)
    expected = torch.nn.functional.gelu(a @ b + bias).to(torch.bfloat16)
    torch.testing.assert_close(out.float(), expected.float(), rtol=3e-2, atol=1.25e-1)
    print(f"PASS bf16_gemm_bias_gelu ({m},{n},{k})")


def _check_reuse_and_reject(ops) -> None:
    a = torch.randn((16, 32), device="cuda", dtype=torch.bfloat16).contiguous()
    b = torch.randn((32, 64), device="cuda", dtype=torch.bfloat16).contiguous()
    bias = torch.randn((64,), device="cuda", dtype=torch.bfloat16).contiguous()
    out = torch.empty((16, 64), device="cuda", dtype=torch.bfloat16)
    returned = ops.gemm_bias_gelu(a, b, bias, out=out)
    if returned is not out:
        raise AssertionError("out tensor must be reused in place")
    try:
        ops.gemm_bias_gelu(a, torch.randn((31, 64), device="cuda", dtype=torch.bfloat16).contiguous(), bias)
    except RuntimeError as exc:
        if "a.shape[1]" not in str(exc):
            raise
    else:
        raise AssertionError("wrong b shape must be rejected")
    try:
        ops.gemm_bias_gelu(a, b, torch.randn((63,), device="cuda", dtype=torch.bfloat16).contiguous())
    except RuntimeError as exc:
        if "bias length" not in str(exc):
            raise
    else:
        raise AssertionError("wrong bias shape must be rejected")
    print("PASS bf16_gemm_bias_gelu reuse + rejections")


def run(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    shapes = [(16, 64, 32)] if args.mode == "smoke" else [(16, 64, 32), (32, 128, 64)]
    for m, n, k in shapes:
        _check_gemm_bias(ops, m, n, k)
        _check_gemm_bias_gelu(ops, m, n, k)
    _check_reuse_and_reject(ops)
    print(f"PASS flashrt-gemm-epilogues/bf16-gemm {args.backend} mode={args.mode}: "
          f"{2 * len(shapes) + 1} checks")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        import traceback
        traceback.print_exc()
        return 1
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps({"passed": 1, "total": 1, "backend": args.backend}) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
