#!/usr/bin/env python3
"""QKV + RoPE epilogue reference benchmark (reference-only package, no CUDA kernel yet).

This package publishes a reference/autograd API only. No speedup is claimed;
the table records reference latency vs an equivalent torch.compile region.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch


def _apply_mem_cap(max_mem_gb: float = 30.0) -> None:
    if not torch.cuda.is_available() or max_mem_gb <= 0:
        return
    total = torch.cuda.get_device_properties(0).total_memory
    cap = int(max_mem_gb * 1024**3)
    if total <= 0 or cap >= total:
        return
    torch.cuda.set_per_process_memory_fraction(cap / total)


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
        return importlib.import_module("flashrt_qkv_epilogue_train")
    if artifact:
        sys.path.insert(0, artifact)
    return importlib.import_module("flashrt_qkv_epilogue_train")


def run_case(ops, label: str, b: int, t: int, hid: int, qh: int, kvh: int, d: int) -> dict:
    torch.manual_seed(0)
    x = torch.randn(b, t, hid, device="cuda", dtype=torch.bfloat16)
    wq = torch.randn(qh * d, hid, device="cuda", dtype=torch.bfloat16)
    wk = torch.randn(kvh * d, hid, device="cuda", dtype=torch.bfloat16)
    wv = torch.randn(kvh * d, hid, device="cuda", dtype=torch.bfloat16)
    cos = torch.randn(b, t, d, device="cuda", dtype=torch.bfloat16)
    sin = torch.randn(b, t, d, device="cuda", dtype=torch.bfloat16)

    def ref():
        return ops.qkv_rope_reference(x, wq, wk, wv, cos, sin, qh, kvh, d)

    ref_us = elapsed_us(ref)
    compiled = torch.compile(ref, mode="reduce-overhead")
    compiled()
    compile_us = elapsed_us(lambda: compiled())
    return {
        "label": label, "B": b, "T": t, "H": hid, "qh": qh, "kvh": kvh, "D": d,
        "reference_us": ref_us, "compile_us": compile_us,
        "ratio": compile_us / ref_us,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="installed")
    parser.add_argument("--artifact")
    parser.add_argument("--max-mem-gb", type=float, default=30.0)
    args = parser.parse_args()
    _apply_mem_cap(args.max_mem_gb)
    ops = load_ops(args.backend, args.artifact)
    print("label,B,T,H,qh,kvh,D,reference_us,compile_us,compile_over_reference")
    for label, b, t, hid, qh, kvh, d in [
        ("qkv_rope_2k", 1, 2048, 4096, 32, 8, 128),
        ("qkv_rope_4k", 1, 4096, 4096, 32, 8, 128),
    ]:
        r = run_case(ops, label, b, t, hid, qh, kvh, d)
        print(
            f"{r['label']},{r['B']},{r['T']},{r['H']},{r['qh']},{r['kvh']},"
            f"{r['D']},{r['reference_us']:.3f},{r['compile_us']:.3f},"
            f"{r['ratio']:.2f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
