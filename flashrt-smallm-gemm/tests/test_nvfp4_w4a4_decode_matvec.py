#!/usr/bin/env python3
"""Correctness tests for flashrt-smallm-gemm (NVFP4 W4A4 M=1 decode matvec)."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "flashrt-smallm-gemm"
REGISTRATION_INCLUDE = (
    ROOT.parent
    / "kernels"
    / "kernel-builder"
    / "src"
    / "pyproject"
    / "templates"
    / "torch"
)


def _swizzled_bytes(rows: int, D: int) -> int:
    n_blocks = D // 16
    return ((rows + 127) // 128) * ((n_blocks + 3) // 4) * 512


def _swizzle_scales(scales: torch.Tensor) -> torch.Tensor:
    rows, n_blocks = scales.shape
    n_col_super = (n_blocks + 3) // 4
    out = torch.zeros(_swizzled_bytes(rows, n_blocks * 16), dtype=torch.uint8)
    src = scales.cpu()
    for row in range(rows):
        rb = row // 128
        ri = row % 128
        for block in range(n_blocks):
            cb = block // 4
            ci = block % 4
            super_idx = rb * n_col_super + cb
            inner_off = (ri % 32) * 16 + (ri // 32) * 4 + ci
            out[super_idx * 512 + inner_off] = src[row, block]
    return out


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

    def decode_matvec(self, a_packed, b_packed, sfa, sfb, *, alpha=1.0, out=None):
        if out is None:
            out = torch.empty((b_packed.shape[0],), device=b_packed.device, dtype=torch.bfloat16)
        self._ops.nvfp4_w4a4_decode_matvec_bf16out(a_packed, b_packed, sfa, sfb, out, float(alpha))
        return out


class InstalledOps:
    def __init__(self, module) -> None:
        self._module = module

    def decode_matvec(self, a_packed, b_packed, sfa, sfb, *, alpha=1.0, out=None):
        return self._module.nvfp4_w4a4_decode_matvec_bf16out(
            a_packed, b_packed, sfa, sfb, alpha=alpha, out=out
        )


def load_source_ops() -> SourceOps:
    from torch.utils.cpp_extension import load

    if not REGISTRATION_INCLUDE.is_dir():
        raise RuntimeError(f"missing kernel-builder registration include: {REGISTRATION_INCLUDE}")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _arch_list())
    namespace = "flashrt_smallm_gemm_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "fp4_w4a4_matvec_sm120.cu"),
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
        return InstalledOps(importlib.import_module("flashrt_smallm_gemm"))
    finally:
        if artifact:
            sys.path.remove(artifact)


def _check_constant_inputs(ops, K: int) -> None:
    torch.manual_seed(0)
    N = 16
    alpha = 0.5
    a_packed = torch.full((K // 2,), 0x11, device="cuda", dtype=torch.uint8)
    b_packed = torch.full((N, K // 2), 0x11, device="cuda", dtype=torch.uint8)
    sfa = _swizzle_scales(torch.full((1, K // 16), 0x38, dtype=torch.uint8)).cuda()
    sfb = _swizzle_scales(torch.full((N, K // 16), 0x38, dtype=torch.uint8)).cuda()

    out = ops.decode_matvec(a_packed, b_packed, sfa, sfb, alpha=alpha)
    expected = torch.full((N,), K * 0.25 * alpha, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(out, expected)
    print(f"PASS w4a4 decode constant K={K}: exact")


def _check_reuse(ops) -> None:
    K = 4096
    N = 8
    a_packed = torch.full((K // 2,), 0x11, device="cuda", dtype=torch.uint8)
    b_packed = torch.full((N, K // 2), 0x11, device="cuda", dtype=torch.uint8)
    sfa = _swizzle_scales(torch.full((1, K // 16), 0x38, dtype=torch.uint8)).cuda()
    sfb = _swizzle_scales(torch.full((N, K // 16), 0x38, dtype=torch.uint8)).cuda()
    out = torch.empty((N,), device="cuda", dtype=torch.bfloat16)
    returned = ops.decode_matvec(a_packed, b_packed, sfa, sfb, out=out)
    if returned is not out:
        raise AssertionError("out tensor must be reused in place")
    print("PASS w4a4 decode reuses out")


def _check_rejects_unsupported_k(ops) -> None:
    K = 8192
    N = 8
    a_packed = torch.full((K // 2,), 0x11, device="cuda", dtype=torch.uint8)
    b_packed = torch.full((N, K // 2), 0x11, device="cuda", dtype=torch.uint8)
    sfa = torch.zeros(_swizzled_bytes(1, K), device="cuda", dtype=torch.uint8)
    sfb = torch.zeros(_swizzled_bytes(N, K), device="cuda", dtype=torch.uint8)
    try:
        ops.decode_matvec(a_packed, b_packed, sfa, sfb)
    except RuntimeError as exc:
        if "K=4096 and K=12288" not in str(exc):
            raise
    else:
        raise AssertionError("K=8192 must be rejected")
    print("PASS w4a4 decode rejects unsupported K")


def run(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    for K in ([4096, 12288] if args.mode == "full" else [4096]):
        _check_constant_inputs(ops, K)
    _check_reuse(ops)
    _check_rejects_unsupported_k(ops)
    print(f"PASS flashrt-smallm-gemm {args.backend} mode={args.mode}: "
          f"{2 + len([4096, 12288] if args.mode == 'full' else [4096])} checks")


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
