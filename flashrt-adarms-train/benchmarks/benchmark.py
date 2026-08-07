#!/usr/bin/env python3
"""AdaRMS / gated-residual AdaRMS training benchmark (kernel vs torch.compile)."""

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
        return importlib.import_module("flashrt_adarms_train")
    if artifact:
        sys.path.insert(0, artifact)
    return importlib.import_module("flashrt_adarms_train")


def run_case(ops, label: str, b: int, t: int, h: int) -> dict:
    torch.manual_seed(0)
    x = torch.randn(b, t, h, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    mod = torch.randn(b, 1, 3 * h, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    xb = torch.randn(b, t, h, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    hb = torch.randn_like(xb, requires_grad=True)
    gb = torch.randn_like(xb, requires_grad=True)
    mb = torch.randn(b, 3 * h, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def fwd():
        y, _g = ops.adarms(x, mod)
        return y

    def fwd_bwd():
        y, _g = ops.adarms(x, mod)
        y.float().sum().backward()
        return y

    def resgate():
        return ops.resgate_adarms(xb, hb, gb, mb)[1]

    def resgate_bwd():
        y = ops.resgate_adarms(xb, hb, gb, mb)[1]
        y.float().sum().backward()
        return y

    fwd_us = elapsed_us(fwd)
    fwd_bwd_us = elapsed_us(fwd_bwd)
    resgate_us = elapsed_us(resgate)
    resgate_bwd_us = elapsed_us(resgate_bwd)

    comp_fwd = torch.compile(fwd, mode="reduce-overhead")
    comp_fwd()
    compile_us = elapsed_us(lambda: comp_fwd())

    comp_rg = torch.compile(resgate, mode="reduce-overhead")
    comp_rg()
    compile_resgate_us = elapsed_us(lambda: comp_rg())

    return {
        "label": label, "B": b, "T": t, "H": h,
        "adarms_fwd_us": fwd_us, "adarms_fwd_bwd_us": fwd_bwd_us,
        "resgate_fwd_us": resgate_us, "resgate_fwd_bwd_us": resgate_bwd_us,
        "compile_fwd_us": compile_us, "compile_resgate_us": compile_resgate_us,
        "vs_compile_fwd": compile_us / fwd_us,
        "vs_compile_resgate": compile_resgate_us / resgate_us,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="installed")
    parser.add_argument("--artifact")
    args = parser.parse_args()
    ops = load_ops(args.backend, args.artifact)
    print("label,B,T,H,adarms_fwd_us,adarms_fwd_bwd_us,resgate_fwd_us,resgate_fwd_bwd_us,compile_fwd_us,compile_resgate_us,vs_compile_fwd,vs_compile_resgate")
    for label, b, t, h in [
        ("vla_1k", 1, 1024, 4096),
        ("vla_2k", 1, 2048, 4096),
        ("vla_4k_b4", 4, 1024, 4096),
    ]:
        r = run_case(ops, label, b, t, h)
        print(
            f"{r['label']},{r['B']},{r['T']},{r['H']},{r['adarms_fwd_us']:.3f},"
            f"{r['adarms_fwd_bwd_us']:.3f},{r['resgate_fwd_us']:.3f},"
            f"{r['resgate_fwd_bwd_us']:.3f},{r['compile_fwd_us']:.3f},"
            f"{r['compile_resgate_us']:.3f},"
            f"{r['vs_compile_fwd']:.2f}x,{r['vs_compile_resgate']:.2f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
