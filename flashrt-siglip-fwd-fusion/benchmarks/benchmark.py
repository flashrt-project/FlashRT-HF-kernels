#!/usr/bin/env python3
"""SigLIP forward-fusion reference benchmark (reference-only package, no CUDA kernel yet).

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
        return importlib.import_module("flashrt_siglip_fwd_fusion")
    if artifact:
        sys.path.insert(0, artifact)
    return importlib.import_module("flashrt_siglip_fwd_fusion")


def run_case(ops, label: str, b: int, t: int, d: int) -> dict:
    torch.manual_seed(0)
    x = torch.randn(b, t, d, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(b, t, d, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(d, device="cuda", dtype=torch.bfloat16) + 1.0
    bias = torch.randn(d, device="cuda", dtype=torch.bfloat16)

    def ln_ref():
        return ops.siglip_residual_layernorm_fwd(x, residual, weight, bias)

    def gelu_ref():
        return ops.siglip_gelu_fwd(x, bias)

    ln_us = elapsed_us(ln_ref)
    gelu_us = elapsed_us(gelu_ref)
    ln_c = torch.compile(ln_ref, mode="reduce-overhead")
    ln_c()
    ln_c_us = elapsed_us(lambda: ln_c())
    gelu_c = torch.compile(gelu_ref, mode="reduce-overhead")
    gelu_c()
    gelu_c_us = elapsed_us(lambda: gelu_c())
    return {
        "label": label, "B": b, "T": t, "D": d,
        "ln_us": ln_us, "ln_compile_us": ln_c_us,
        "gelu_us": gelu_us, "gelu_compile_us": gelu_c_us,
        "ln_ratio": ln_c_us / ln_us,
        "gelu_ratio": gelu_c_us / gelu_us,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="installed")
    parser.add_argument("--artifact")
    args = parser.parse_args()
    ops = load_ops(args.backend, args.artifact)
    print("label,B,T,D,ln_us,ln_compile_us,gelu_us,gelu_compile_us,ln_compile_over_ref,gelu_compile_over_ref")
    for label, b, t, d in [
        ("siglip_1k", 1, 1024, 768),
        ("siglip_2k", 2, 1024, 1024),
    ]:
        r = run_case(ops, label, b, t, d)
        print(
            f"{r['label']},{r['B']},{r['T']},{r['D']},{r['ln_us']:.3f},"
            f"{r['ln_compile_us']:.3f},{r['gelu_us']:.3f},{r['gelu_compile_us']:.3f},"
            f"{r['ln_ratio']:.2f}x,{r['gelu_ratio']:.2f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
