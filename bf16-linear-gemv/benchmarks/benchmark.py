#!/usr/bin/env python3
"""BF16 M=1 decode GEMV benchmark (kernel vs eager vs torch.compile)."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch


def elapsed_us(fn, warmup: int = 20, repeats: int = 100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / repeats


def load_ops(backend: str, artifact: str | None):
    if backend == "source":
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "torch-ext"))
        return importlib.import_module("bf16_linear_gemv")
    if artifact:
        sys.path.insert(0, artifact)
    return importlib.import_module("bf16_linear_gemv")


def run_case(ops, label: str, n: int, k: int) -> dict:
    gen = torch.Generator(device="cuda").manual_seed(0)
    x = (torch.randn((k,), device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
    w = (torch.randn((n, k), device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
    out = torch.empty((n,), device="cuda", dtype=torch.bfloat16)
    eager = lambda: x.float() @ w.float().t()
    compiled = torch.compile(eager, mode="reduce-overhead")
    compiled()

    def kernel_v0():
        ops.bf16_decode_gemv_bf16(x, w, out=out)

    def kernel_v1():
        ops.bf16_decode_gemv_bf16(x, w, variant=16, out=out)

    def kernel_unrolled():
        ops.bf16_decode_gemv_unrolled_bf16(x, w, out=out)

    kernel_us = elapsed_us(kernel_v0)
    kernel_v1_us = elapsed_us(kernel_v1)
    kernel_ur_us = elapsed_us(kernel_unrolled)
    eager_us = elapsed_us(eager)
    compile_us = elapsed_us(lambda: compiled())
    return {
        "label": label, "N": n, "K": k,
        "kernel_v0_us": kernel_us, "kernel_v1_us": kernel_v1_us,
        "kernel_unrolled_us": kernel_ur_us, "eager_us": eager_us,
        "compile_us": compile_us,
        "vs_eager_v0": eager_us / kernel_us,
        "vs_eager_v1": eager_us / kernel_v1_us,
        "vs_eager_ur": eager_us / kernel_ur_us,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="installed")
    parser.add_argument("--artifact")
    args = parser.parse_args()
    ops = load_ops(args.backend, args.artifact)
    print("label,N,K,kernel_v0_us,kernel_v1_us,kernel_unrolled_us,eager_us,compile_us,vs_eager_v0,vs_eager_v1,vs_eager_ur")
    for label, n, k in [
        ("decode_8k", 8192, 4096),
        ("decode_12k", 12288, 8192),
        ("decode_16k", 16384, 8192),
    ]:
        r = run_case(ops, label, n, k)
        print(
            f"{r['label']},{r['N']},{r['K']},{r['kernel_v0_us']:.3f},"
            f"{r['kernel_v1_us']:.3f},{r['kernel_unrolled_us']:.3f},"
            f"{r['eager_us']:.3f},{r['compile_us']:.3f},"
            f"{r['vs_eager_v0']:.2f}x,{r['vs_eager_v1']:.2f}x,{r['vs_eager_ur']:.2f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
