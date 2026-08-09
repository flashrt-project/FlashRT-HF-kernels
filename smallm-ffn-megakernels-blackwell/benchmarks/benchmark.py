#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "tests"))
from _source_loader import load_installed_ops, load_source_ops  # noqa: E402


def _apply_mem_cap(max_mem_gb: float = 30.0) -> None:
    if not torch.cuda.is_available() or max_mem_gb <= 0:
        return
    total = torch.cuda.get_device_properties(0).total_memory
    cap = int(max_mem_gb * 1024**3)
    if total <= 0 or cap >= total:
        return
    torch.cuda.set_per_process_memory_fraction(cap / total)


F8 = torch.float8_e4m3fn


def fp8(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(-448, 448).to(F8)


def measure(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("source", "installed"), default="source")
    parser.add_argument("--artifact")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--max-mem-gb", type=float, default=30.0)
    args = parser.parse_args()
    _apply_mem_cap(args.max_mem_gb)
    if args.backend == "installed":
        if not args.artifact:
            parser.error("--artifact is required for --backend installed")
        ops = load_installed_ops(args.artifact)
    else:
        ops = load_source_ops(None)

    torch.manual_seed(23)
    print("| Region | M | Kernel us | Exact eager us | Speedup |")
    print("|---|---:|---:|---:|---:|")
    for m in (1, 8, 21, 32):
        x = fp8(torch.randn(m, 1024, device="cuda") * 0.2)
        uw = fp8(torch.randn(4096, 1024, device="cuda") * 0.02)
        dw = fp8(torch.randn(1024, 4096, device="cuda") * 0.02)
        ub = (torch.randn(4096, device="cuda") * 0.01).bfloat16()
        db = (torch.randn(1024, device="cuda") * 0.01).bfloat16()
        dinv = torch.ones(4096, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn(m, 1024, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn_like(gate)
        out = torch.empty_like(gate)
        scratch = torch.empty(m, 4096, device="cuda", dtype=F8)
        kernel = lambda: ops.gated(
            x, uw, ub, dinv, dw, db, gate, residual,
            1.0, 1.0, 1.0, out, scratch
        )
        eager = lambda: (
            fp8(torch.nn.functional.gelu(
                x.float() @ uw.float().T + ub.float(), approximate="tanh"
            )).float() @ dw.float().T + db.float()
        ) * gate.float() + residual.float()
        kt = measure(kernel, args.warmup, args.iterations)
        et = measure(eager, args.warmup, args.iterations)
        print(f"| gated 1024/4096 | {m} | {kt:.3f} | {et:.3f} | {et / kt:.2f}x |")

    for m in (1, 51, 144, 188):
        x = torch.randn(m, 512, device="cuda", dtype=torch.bfloat16) * 0.2
        uw = fp8(torch.randn(2048, 512, device="cuda") * 0.02)
        dw = fp8(torch.randn(512, 2048, device="cuda") * 0.02)
        ub = (torch.randn(2048, device="cuda") * 0.01).bfloat16()
        db = (torch.randn(512, device="cuda") * 0.01).bfloat16()
        uinv = torch.ones(512, device="cuda", dtype=torch.bfloat16)
        dinv = torch.ones(2048, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn(m, 512, device="cuda", dtype=torch.bfloat16)
        out = torch.empty_like(residual)
        xs = torch.empty(m, 512, device="cuda", dtype=F8)
        hs = torch.empty(m, 2048, device="cuda", dtype=F8)
        barrier = torch.zeros(2, device="cuda", dtype=torch.uint32)
        split = torch.cuda.get_device_capability() != (11, 0) and m == 188
        kernel = lambda: ops.residual(
            x, uinv, uw, ub, dinv, dw, db, residual,
            1.0, 1.0, 1.0, 1.0, split, out, xs, hs, barrier
        )
        eager = lambda: fp8(torch.nn.functional.gelu(
            fp8(x.float()).float() @ uw.float().T + ub.float(),
            approximate="tanh",
        )).float() @ dw.float().T + db.float() + residual.float()
        kt = measure(kernel, args.warmup, args.iterations)
        et = measure(eager, args.warmup, args.iterations)
        print(f"| residual 512/2048 | {m} | {kt:.3f} | {et:.3f} | {et / kt:.2f}x |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
