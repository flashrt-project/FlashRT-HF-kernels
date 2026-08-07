#!/usr/bin/env python3
"""RoPE training reference benchmark (reference-only package, no CUDA kernel yet).

No speedup is claimed; the table records reference latency vs an equivalent
torch.compile region.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch


def elapsed_us(fn, warmup: int = 10, repeats: int = 50) -> float:
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
        return importlib.import_module("flashrt_rope_train")
    if artifact:
        sys.path.insert(0, artifact)
    return importlib.import_module("flashrt_rope_train")


def run_case(ops, label: str, b: int, h: int, t: int, d: int) -> dict:
    torch.manual_seed(0)
    q = torch.randn(b, h, t, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, h, t, d, device="cuda", dtype=torch.bfloat16)
    cos = torch.randn(t, d, device="cuda", dtype=torch.bfloat16)
    sin = torch.randn(t, d, device="cuda", dtype=torch.bfloat16)

    def ref():
        return ops.apply_rope_train(q, k, cos, sin)

    ref_us = elapsed_us(ref)
    compiled = torch.compile(ref, mode="reduce-overhead")
    compiled()
    compile_us = elapsed_us(lambda: compiled())
    return {
        "label": label, "B": b, "H": h, "T": t, "D": d,
        "reference_us": ref_us, "compile_us": compile_us,
        "ratio": compile_us / ref_us,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="installed")
    parser.add_argument("--artifact")
    args = parser.parse_args()
    ops = load_ops(args.backend, args.artifact)
    print("label,B,H,T,D,reference_us,compile_us,compile_over_reference")
    for label, b, h, t, d in [
        ("rope_2k", 1, 32, 2048, 128),
        ("rope_4k", 1, 32, 4096, 128),
    ]:
        r = run_case(ops, label, b, h, t, d)
        print(
            f"{r['label']},{r['B']},{r['H']},{r['T']},{r['D']},"
            f"{r['reference_us']:.3f},{r['compile_us']:.3f},{r['ratio']:.2f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
