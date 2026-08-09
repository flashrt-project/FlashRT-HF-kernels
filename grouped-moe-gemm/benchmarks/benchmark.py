#!/usr/bin/env python3
"""35B-A3B grouped expert prefill GEMM package/native parity benchmark."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch


PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "tests"))
from _source_loader import load_source_ops  # noqa: E402
from test_grouped_moe_gemm import make_sf  # noqa: E402


WORKLOADS = {
    "35b_gate_up": (64, 1024, 2048, 16, 256),
    "35b_down": (64, 2048, 512, 16, 256),
}


def _apply_mem_cap(max_mem_gb: float = 30.0) -> None:
    if not torch.cuda.is_available() or max_mem_gb <= 0:
        return
    total = torch.cuda.get_device_properties(0).total_memory
    cap = int(max_mem_gb * 1024**3)
    if total <= 0 or cap >= total:
        return
    torch.cuda.set_per_process_memory_fraction(cap / total)


def time_us(fn, warmup: int, iters: int) -> float:
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


def load_ops(backend: str, artifact: str | None):
    if backend == "source":
        return load_source_ops()
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return importlib.import_module("grouped_moe_gemm")
    finally:
        if artifact:
            sys.path.remove(artifact)


def make_case(tile_rows: int, n: int, k: int, tiles: int, experts: int):
    rows = tile_rows * tiles
    dev = "cuda"
    packed = torch.randint(0, 256, (rows, k // 2), device=dev, dtype=torch.uint8)
    weights = torch.randint(
        0, 256, (experts, n, k // 2), device=dev, dtype=torch.uint8
    )
    input_scale, _ = make_sf(rows, k, dev)
    weight_scales = torch.stack([make_sf(n, k, dev)[0] for _ in range(experts)])
    alpha = torch.rand(experts, device=dev, dtype=torch.float32) + 0.5
    tile_expert = torch.arange(tiles, device=dev, dtype=torch.int32) % experts
    out = torch.empty(rows, n, device=dev, dtype=torch.bfloat16)
    return packed, weights, input_scale, weight_scales, alpha, tile_expert, out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("source", "installed"), default="source")
    parser.add_argument("--artifact")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--max-mem-gb", type=float, default=30.0)
    parser.add_argument("--json-out")
    args = parser.parse_args()
    _apply_mem_cap(args.max_mem_gb)

    torch.manual_seed(9102)
    ops = load_ops(args.backend, args.artifact)
    rows = []
    for name, (tile_rows, n, k, tiles, experts) in WORKLOADS.items():
        packed, weights, input_scale, weight_scales, alpha, tile_expert, out = make_case(
            tile_rows, n, k, tiles, experts
        )
        kwargs = dict(
            tile_rows=tile_rows,
            weight_stride=weights[0].numel(),
            weight_scale_stride=weight_scales[0].numel(),
        )

        def public_call():
            return ops.grouped_nvfp4_gemm_bf16(
                packed, weights, input_scale, weight_scales, alpha,
                tile_expert, out=out, **kwargs
            )

        def raw_call():
            ops.ops.grouped_nvfp4_gemm_bf16_out(
                packed, weights, input_scale, weight_scales, alpha,
                tile_expert, tile_rows, 0, kwargs["weight_stride"],
                kwargs["weight_scale_stride"], out,
            )

        expected = public_call().clone()
        raw_call()
        torch.cuda.synchronize()
        if not torch.equal(out, expected):
            raise AssertionError(f"{name}: public/raw output mismatch")
        public_us = time_us(public_call, args.warmup, args.iters)
        raw_us = time_us(raw_call, args.warmup, args.iters)
        ratio = public_us / raw_us
        accepted = public_us - raw_us <= max(0.75, raw_us * 0.05)
        row = {
            "workload": name,
            "tile_rows": tile_rows,
            "rows": tile_rows * tiles,
            "n": n,
            "k": k,
            "experts": experts,
            "raw_native_us": raw_us,
            "public_wrapper_us": public_us,
            "wrapper_native": ratio,
            "bit_exact": True,
            "accepted": accepted,
        }
        rows.append(row)
        print(
            f"{name}: raw={raw_us:.3f}us public={public_us:.3f}us "
            f"ratio={ratio:.4f} accepted={accepted}"
        )
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(rows, indent=2) + "\n")
    if not all(row["accepted"] for row in rows):
        raise AssertionError("grouped MoE wrapper/native parity failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
