#!/usr/bin/env python3
"""Correctness tests for grouped-moe-gemv."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "grouped-moe-gemv"
REGISTRATION_INCLUDE = ROOT.parent / "kernels" / "kernel-builder" / "src" / "pyproject" / "templates" / "torch"


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self.ops = getattr(torch.ops, namespace)

    def w4a16_decode_gemv_bf16(self, x, w, sfb, alpha=1.0):
        out = torch.empty((w.shape[0],), device=x.device, dtype=torch.bfloat16)
        self.ops.w4a16_decode_gemv_bf16(x, w, sfb, float(alpha), out)
        return out

    def grouped_w4a16_gemv_bf16(self, acts, w_stack, sfb_stack, alpha_stack, expert_idx, w_stride, sfb_stride, n):
        out = torch.empty((acts.shape[0], n), device=acts.device, dtype=torch.bfloat16)
        self.ops.grouped_w4a16_gemv_bf16(acts, w_stack, sfb_stack, alpha_stack, expert_idx, int(w_stride), int(sfb_stride), out)
        return out


def _arch_list() -> str:
    major, minor = torch.cuda.get_device_capability(0)
    return "12.0a" if major >= 12 else f"{major}.{minor}"


def load_source_ops() -> SourceOps:
    from torch.utils.cpp_extension import load

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _arch_list())
    namespace = "grouped_moe_gemv_source_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "kernels" / "nexn2_w4a16_gemv.cu"),
            str(PACKAGE / "csrc" / "kernels" / "nexn2_moe_grouped_w4a16.cu"),
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
        return importlib.import_module("grouped_moe_gemv")
    finally:
        if artifact:
            sys.path.remove(artifact)


def sfb_bytes(rows: int, k: int) -> int:
    n_blocks = k // 16
    n_row_super = (rows + 127) // 128
    n_col_super = (n_blocks + 3) // 4
    return n_row_super * n_col_super * 512


def assert_constant(name: str, got: torch.Tensor, expected: float) -> None:
    ref = torch.full_like(got.float(), expected)
    diff = (got.float() - ref).abs()
    max_abs = float(diff.max().item())
    if max_abs > 0.0:
        raise AssertionError(f"{name}: expected exact {expected}, max_abs={max_abs}")


def run(ops, mode: str) -> int:
    shapes = [(64, 128), (128, 256)] if mode == "smoke" else [(64, 128), (128, 256), (256, 512)]
    count = 0
    for n, k in shapes:
        x = torch.ones((k,), device="cuda", dtype=torch.bfloat16)
        packed = torch.full((n, k // 2), 0x11, device="cuda", dtype=torch.uint8)
        sfb = torch.full((sfb_bytes(n, k),), 0x38, device="cuda", dtype=torch.uint8)
        got = ops.w4a16_decode_gemv_bf16(x, packed, sfb, alpha=1.0)
        torch.cuda.synchronize()
        assert_constant(f"w4a16_decode n={n} k={k}", got, k * 0.5)
        count += 1

        slots = 4
        experts = 3
        acts = torch.ones((slots, k), device="cuda", dtype=torch.bfloat16)
        w_stack = torch.full((experts, n, k // 2), 0x11, device="cuda", dtype=torch.uint8).contiguous()
        sfb_one = sfb_bytes(n, k)
        sfb_stack = torch.full((experts, sfb_one), 0x38, device="cuda", dtype=torch.uint8).contiguous()
        alpha = torch.tensor([1.0, 0.5, 2.0], device="cuda", dtype=torch.float32)
        expert_idx = torch.tensor([0, 1, 2, 1], device="cuda", dtype=torch.int32)
        got_g = ops.grouped_w4a16_gemv_bf16(
            acts, w_stack, sfb_stack, alpha, expert_idx,
            w_stride=n * k // 2, sfb_stride=sfb_one, n=n)
        torch.cuda.synchronize()
        expected = torch.tensor([k * 0.5, k * 0.25, k * 1.0, k * 0.25], device="cuda", dtype=torch.float32)[:, None]
        diff = (got_g.float() - expected).abs()
        if float(diff.max().item()) > 0.0:
            raise AssertionError("grouped_w4a16_gemv_bf16 constant mismatch")
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    count = run(ops, args.mode)
    print(f"grouped-moe-gemv {args.backend} {args.mode}: passed {count}/{count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
