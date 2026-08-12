#!/usr/bin/env python3
"""Correctness tests for flashrt-gemm-epilogues BF16 linear (bias) ops."""
from __future__ import annotations

import argparse
import importlib
import json
import math
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

LINEAR_SHAPES = [
    ("decode_m1_1024", 1, 1024, 1024),
    ("decode_m1_qkv", 1, 1024, 2560),
    ("decode_m8_1024", 8, 1024, 1024),
    ("decode_m10_qkv", 10, 1024, 2560),
    ("vlm_m512_square", 512, 1152, 1152),
    ("vla_m1024_square", 1024, 2048, 2048),
]

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

    def linear(self, x, w, *, out=None):
        if out is None:
            out = torch.empty((x.shape[0], w.shape[1]), device=x.device, dtype=torch.bfloat16)
        self._ops.bf16_linear_bf16(x, w, out)
        return out

    def linear_bias(self, x, w, bias, *, out=None):
        if out is None:
            out = torch.empty((x.shape[0], w.shape[1]), device=x.device, dtype=torch.bfloat16)
        self._ops.bf16_linear_bias_bf16(x, w, bias, out)
        return out


class InstalledOps:
    def __init__(self, module) -> None:
        self._module = module

    def linear(self, x, w, *, out=None):
        return self._module.bf16_linear_bf16(x, w, out=out)

    def linear_bias(self, x, w, bias, *, out=None):
        return self._module.bf16_linear_bias_bf16(x, w, bias, out=out)


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


def _percentile(x: torch.Tensor, q: float) -> torch.Tensor:
    flat = x.flatten()
    k = max(1, min(flat.numel(), math.ceil(q * flat.numel())))
    return flat.kthvalue(k).values


def _check_linear(ops, name: str, m: int, k: int, n: int) -> None:
    torch.manual_seed(11)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16).contiguous()
    w = torch.randn((k, n), device="cuda", dtype=torch.bfloat16).contiguous()
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    returned = ops.linear(x, w, out=out)
    if returned is not out:
        raise AssertionError("out tensor must be reused in place")
    expected = (x @ w).to(torch.bfloat16)
    diff = (out.float() - expected.float()).abs()
    p99 = float(_percentile(diff, 0.99).item())
    cosine = float(torch.nn.functional.cosine_similarity(out.float().flatten(), expected.float().flatten(), dim=0).item())
    if not (p99 <= 0.5 and cosine >= 0.999):
        raise AssertionError(f"{name} linear failed: p99={p99} cosine={cosine}")
    print(f"PASS bf16_linear {name} ({m},{k},{n}): p99={p99:.6f} cosine={cosine:.8f}")


def _check_linear_bias(ops, name: str, m: int, k: int, n: int) -> None:
    torch.manual_seed(17)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16).contiguous()
    w = torch.randn((k, n), device="cuda", dtype=torch.bfloat16).contiguous()
    bias = torch.randn((n,), device="cuda", dtype=torch.bfloat16).contiguous()
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    returned = ops.linear_bias(x, w, bias, out=out)
    if returned is not out:
        raise AssertionError("out tensor must be reused in place")
    expected = torch.addmm(bias, x, w).to(torch.bfloat16)
    diff = (out.float() - expected.float()).abs()
    p99 = float(_percentile(diff, 0.99).item())
    cosine = float(torch.nn.functional.cosine_similarity(out.float().flatten(), expected.float().flatten(), dim=0).item())
    if not (p99 <= 0.5 and cosine >= 0.999):
        raise AssertionError(f"{name} linear_bias failed: p99={p99} cosine={cosine}")
    print(f"PASS bf16_linear_bias {name} ({m},{k},{n}): p99={p99:.6f} cosine={cosine:.8f}")


def _check_rejections(ops) -> None:
    x = torch.randn((10, 32), device="cuda", dtype=torch.bfloat16).contiguous()
    w_bad = torch.randn((31, 1024), device="cuda", dtype=torch.bfloat16).contiguous()
    try:
        ops.linear(x, w_bad)
    except RuntimeError as exc:
        if "x.shape[1]" not in str(exc):
            raise
    else:
        raise AssertionError("wrong w shape must be rejected")
    bias_bad = torch.randn((1023,), device="cuda", dtype=torch.bfloat16).contiguous()
    w = torch.randn((32, 1024), device="cuda", dtype=torch.bfloat16).contiguous()
    try:
        ops.linear_bias(x, w, bias_bad)
    except RuntimeError as exc:
        if "bias length" not in str(exc):
            raise
    else:
        raise AssertionError("wrong bias shape must be rejected")
    try:
        ops.linear(torch.randn((32, 10), device="cuda", dtype=torch.bfloat16).t().contiguous() if False else torch.randn((32, 10), device="cuda", dtype=torch.bfloat16).t(), w)
    except RuntimeError as exc:
        if "contiguous" not in str(exc):
            raise
    else:
        raise AssertionError("non-contiguous x must be rejected")
    print("PASS bf16_linear rejections")


def run(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    shapes = LINEAR_SHAPES if args.mode == "full" else LINEAR_SHAPES[:3]
    for name, m, k, n in shapes:
        _check_linear(ops, name, m, k, n)
        _check_linear_bias(ops, name, m, k, n)
    _check_rejections(ops)
    print(f"PASS flashrt-gemm-epilogues/bf16-linear {args.backend} mode={args.mode}: "
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
