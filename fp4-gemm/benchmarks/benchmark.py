#!/usr/bin/env python3
"""Benchmark fp4-gemm."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
TEST_FILE = ROOT / "fp4-gemm" / "tests" / "test_fp4_gemm.py"


@dataclass
class BenchResult:
    shape: str
    M: int
    N: int
    K: int
    variant: int
    flashrt_us: float
    torch_reference_us: float
    speedup_vs_reference: float
    max_abs: float
    mean_abs: float
    p99_abs: float
    cosine: float
    status: str


def load_helpers():
    spec = importlib.util.spec_from_file_location("fp4_gemm_test_helpers", TEST_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helpers from {TEST_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["fp4_gemm_test_helpers"] = module
    spec.loader.exec_module(module)
    return module


def measure(fn, warmup: int, iters: int) -> float:
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
    return float(start.elapsed_time(end) * 1000.0 / iters)


def bench_case(helpers, ops, name: str, shape: tuple[int, int, int], warmup: int, iters: int) -> list[BenchResult]:
    m, n, k = shape
    a_packed, b_packed, sfa, sfb, expected = helpers.prepare_quantized(ops, m, n, k)
    a_deq = torch.empty((m, k), device="cuda", dtype=torch.float16)
    b_deq = torch.empty((n, k), device="cuda", dtype=torch.float16)
    ops.dequantize_fp4_sfa_fp16(a_packed, sfa, a_deq, False)
    ops.dequantize_fp4_sfa_fp16(b_packed, sfb, b_deq, True)
    torch.cuda.synchronize()

    def torch_ref():
        return (a_deq.float() @ b_deq.float().T).to(torch.bfloat16)

    torch_us = measure(torch_ref, warmup, iters)
    results: list[BenchResult] = []
    for variant in (0, 1, 2):
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        ops.fp4_w4a16_linear_bf16(a_packed, b_packed, sfa, sfb, out, 1.0, variant)
        torch.cuda.synchronize()
        max_abs, mean_abs, p99_abs, cosine = helpers.metrics(out, expected)
        flashrt_us = measure(
            lambda: ops.fp4_w4a16_linear_bf16(a_packed, b_packed, sfa, sfb, out, 1.0, variant),
            warmup,
            iters,
        )
        results.append(
            BenchResult(
                shape=name,
                M=m,
                N=n,
                K=k,
                variant=variant,
                flashrt_us=flashrt_us,
                torch_reference_us=torch_us,
                speedup_vs_reference=torch_us / flashrt_us,
                max_abs=max_abs,
                mean_abs=mean_abs,
                p99_abs=p99_abs,
                cosine=cosine,
                status="ok",
            )
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "headline"], default="headline")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    helpers = load_helpers()
    ops = helpers.load_source_ops()
    shapes = {
        "small_m16_n128_k128": (16, 128, 128),
        "small_m32_n256_k256": (32, 256, 256),
        "mlp_tile_m64_n512_k512": (64, 512, 512),
    }
    if args.mode == "smoke":
        shapes = {"small_m16_n128_k128": shapes["small_m16_n128_k128"]}
    results: list[BenchResult] = []
    for name, shape in shapes.items():
        results.extend(bench_case(helpers, ops, name, shape, args.warmup, args.iterations))
    payload = {
        "mode": args.mode,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "results": [asdict(item) for item in results],
    }
    print(json.dumps(payload, indent=2))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
