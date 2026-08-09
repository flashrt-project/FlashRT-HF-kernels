#!/usr/bin/env python3
"""Grouped NVFP4 MoE GEMM benchmark (kernel-call latency, SM110 SIMT path)."""

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


def elapsed_us(fn, warmup: int = 20, repeats: int = 50) -> float:
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
        return importlib.import_module("grouped_moe_gemm")
    if artifact:
        sys.path.insert(0, artifact)
    return importlib.import_module("grouped_moe_gemm")


def sf_bytes(rows: int, k: int) -> int:
    return ((rows + 127) // 128) * (((k // 16) + 3) // 4) * 512


def run_case(ops, label: str, tile_rows: int, rows: int, n: int, k: int, tiles: int, experts: int) -> dict:
    ap = torch.randint(0, 256, (rows, k // 2), device="cuda", dtype=torch.uint8)
    wp = torch.randint(0, 256, (experts, n, k // 2), device="cuda", dtype=torch.uint8)
    asw = torch.zeros(rows, sf_bytes(rows, k), device="cuda", dtype=torch.uint8)
    wsw = torch.zeros(experts, sf_bytes(n, k), device="cuda", dtype=torch.uint8)
    alpha = torch.rand(experts, device="cuda") + 0.5
    te = torch.arange(tiles, device="cuda", dtype=torch.int32) % experts

    def kernel():
        ops.grouped_nvfp4_gemm_bf16(
            ap, wp, asw, wsw, alpha, te, tile_rows=tile_rows
        )

    us = elapsed_us(kernel)
    return {"label": label, "tile": tile_rows, "rows": rows, "N": n, "K": k, "tiles": tiles, "experts": experts, "us": us}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="installed")
    parser.add_argument("--artifact")
    parser.add_argument("--max-mem-gb", type=float, default=30.0)
    args = parser.parse_args()
    _apply_mem_cap(args.max_mem_gb)
    ops = load_ops(args.backend, args.artifact)
    print("label,tile_rows,rows,N,K,tiles,experts,latency_us")
    for cfg in [
        ("gate_up_t16", 16, 1024, 1024, 2048, 64, 8),
        ("gate_up_t64", 64, 1024, 1024, 2048, 16, 8),
        ("gate_up_wide_t64", 64, 1024, 4096, 2048, 16, 8),
    ]:
        r = run_case(ops, *cfg)
        print(
            f"{r['label']},{r['tile']},{r['rows']},{r['N']},{r['K']},"
            f"{r['tiles']},{r['experts']},{r['us']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
