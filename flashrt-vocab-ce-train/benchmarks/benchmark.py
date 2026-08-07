#!/usr/bin/env python3
"""Fused linear-CE streaming-kernel benchmark (kernel vs reference vs torch.compile)."""

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
        return importlib.import_module("flashrt_vocab_ce_train")
    if artifact:
        sys.path.insert(0, artifact)
    return importlib.import_module("flashrt_vocab_ce_train")


def run_case(ops, label: str, n: int, h: int, v: int) -> dict:
    torch.manual_seed(0)
    x = torch.randn(n, h, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(v, h, device="cuda", dtype=torch.float32) * 0.05
    labels = torch.randint(0, v, (n,), device="cuda", dtype=torch.int64)
    labels[0] = -100

    def kernel():
        return ops.vocab_ce(x, w, labels, 0.01)

    def reference():
        return ops.reference_vocab_ce(x, w, labels, 0.01)

    kernel_us = elapsed_us(kernel)
    ref_us = elapsed_us(reference)

    compiled = torch.compile(reference, mode="reduce-overhead")
    compiled()
    compile_us = elapsed_us(lambda: compiled())

    return {
        "label": label, "N": n, "H": h, "V": v,
        "kernel_us": kernel_us, "reference_us": ref_us,
        "compile_us": compile_us,
        "vs_reference": ref_us / kernel_us,
        "vs_compile": compile_us / kernel_us,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="installed")
    parser.add_argument("--artifact")
    args = parser.parse_args()
    ops = load_ops(args.backend, args.artifact)
    print("label,N,H,V,kernel_us,reference_us,compile_us,vs_reference,vs_compile")
    for label, n, h, v in [
        ("small_n", 16, 2048, 65536),
        ("mid_n", 64, 2048, 257152),
        ("max_n", 128, 2048, 257152),
    ]:
        r = run_case(ops, label, n, h, v)
        print(
            f"{r['label']},{r['N']},{r['H']},{r['V']},{r['kernel_us']:.3f},"
            f"{r['reference_us']:.3f},{r['compile_us']:.3f},"
            f"{r['vs_reference']:.2f}x,{r['vs_compile']:.2f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
