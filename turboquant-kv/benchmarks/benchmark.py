#!/usr/bin/env python3
"""Benchmark turboquant-kv source or installed artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "tests"))
from test_turboquant_kv import load_installed_ops, load_source_ops, ref_unpack  # noqa: E402


def bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def make_packed(m: int):
    k_idx = torch.randint(0, 256, (m, 128), device="cuda", dtype=torch.uint8)
    k_qjl = torch.randint(0, 256, (m, 32), device="cuda", dtype=torch.uint8)
    v_idx = torch.randint(0, 256, (m, 128), device="cuda", dtype=torch.uint8)
    cb_k = torch.randn((16,), device="cuda", dtype=torch.float32)
    cb_v = torch.randn((16,), device="cuda", dtype=torch.float32)
    return k_idx, k_qjl, v_idx, cb_k, cb_v


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(1234)
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)

    print("| Workload | Shape | FlashRT us | PyTorch eager us | Speedup |")
    print("|---|---:|---:|---:|---:|")

    for m in [1, 4, 128, 1024, 4096, 32768]:
        k_idx, k_qjl, v_idx, cb_k, cb_v = make_packed(m)
        b_k, b_v = 3, 4
        fused = bench(
            lambda: ops.unpack_packed_bf16(k_idx, k_qjl, v_idx, cb_k, cb_v, b_k, b_v),
            args.warmup,
            args.iters,
        )
        eager = bench(
            lambda: ref_unpack(k_idx, k_qjl, v_idx, cb_k, cb_v, b_k, b_v, torch.bfloat16),
            args.warmup,
            args.iters,
        )
        print(f"| unpack_packed_bf16 | M={m}, D=256, bits=3/4 | {fused:.3f} | {eager:.3f} | {eager / fused:.2f}x |")

        fused = bench(
            lambda: ops.unpack_packed_mixed(k_idx, k_qjl, v_idx, cb_k, cb_v, b_k, b_v),
            args.warmup,
            args.iters,
        )
        eager = bench(
            lambda: ref_unpack(k_idx, k_qjl, v_idx, cb_k, cb_v, b_k, b_v, torch.float32),
            args.warmup,
            args.iters,
        )
        print(f"| unpack_packed_mixed | M={m}, D=256, bits=3/4 | {fused:.3f} | {eager:.3f} | {eager / fused:.2f}x |")

    for m in [1, 4, 128, 1024, 4096, 32768]:
        k_mse = torch.randn((m, 256), device="cuda", dtype=torch.bfloat16)
        k_qjl = torch.randn((m, 256), device="cuda", dtype=torch.bfloat16)
        v_unit = torch.randn((m, 256), device="cuda", dtype=torch.bfloat16)
        k_norm = torch.rand((m,), device="cuda", dtype=torch.float16) + 0.5
        k_rnorm = torch.rand((m,), device="cuda", dtype=torch.float16) + 0.5
        v_norm = torch.rand((m,), device="cuda", dtype=torch.float16) + 0.5
        coef = 0.125
        fused = bench(
            lambda: ops.combine_kv_bf16(k_mse, k_qjl, v_unit, k_norm, k_rnorm, v_norm, coef),
            args.warmup,
            args.iters,
        )
        eager = bench(
            lambda: (
                (k_norm.float().unsqueeze(1) * (k_mse.float() + coef * k_rnorm.float().unsqueeze(1) * k_qjl.float())).to(torch.bfloat16),
                (v_norm.float().unsqueeze(1) * v_unit.float()).to(torch.bfloat16),
            ),
            args.warmup,
            args.iters,
        )
        print(f"| combine_kv_bf16 | M={m}, D=256 | {fused:.3f} | {eager:.3f} | {eager / fused:.2f}x |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
