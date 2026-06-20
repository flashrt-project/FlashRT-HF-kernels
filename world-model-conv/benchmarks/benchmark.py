#!/usr/bin/env python3
"""Benchmark world-model-conv against PyTorch eager conv3d references."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "tests"))
from test_world_model_conv import load_installed_ops, load_source_ops, ref_conv  # noqa: E402


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(123)
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    print("| Workload | Shape | FlashRT us | PyTorch eager us | Speedup |")
    print("|---|---:|---:|---:|---:|")
    for shape in [(1, 2, 4, 16, 16, 32, 32), (2, 2, 4, 16, 24, 64, 32), (1, 2, 8, 32, 32, 64, 64)]:
        n, tc, tn, h, w, ci, co = shape
        cache = (torch.randn((n, tc, h, w, ci), device="cuda") * 0.1).to(torch.float8_e4m3fn)
        new = (torch.randn((n, tn, h, w, ci), device="cuda") * 0.1).to(torch.float8_e4m3fn)
        weight = (torch.randn((co, 3, 3, 3, ci), device="cuda") * 0.1).to(torch.float8_e4m3fn)
        bias = (torch.randn((co,), device="cuda") * 0.01).to(torch.bfloat16)
        residual = (torch.randn((n, co, tn, h, w), device="cuda") * 0.05).to(torch.bfloat16)
        alpha = 0.75
        fused = bench(lambda: ops.fp8_conv3d_v18_ncdhw_res_bf16out(cache, new, weight, bias, residual, alpha), args.warmup, args.iters)
        eager = bench(lambda: ref_conv(cache, new, weight, bias, residual, alpha), args.warmup, args.iters)
        print(f"| fp8_conv3d_v18 | N={n},Tc={tc},T={tn},H={h},W={w},Ci={ci},Co={co} | {fused:.3f} | {eager:.3f} | {eager / fused:.2f}x |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
