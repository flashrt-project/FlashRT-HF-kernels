#!/usr/bin/env python3
"""Benchmark small FP32 Cholesky against preallocated PyTorch POTRF."""

from __future__ import annotations

import argparse
import importlib
import math
import statistics
import sys
from pathlib import Path

import torch

TESTS = Path(__file__).resolve().parents[1] / "tests"
sys.path.insert(0, str(TESTS))
from _source_loader import load_source_ops  # noqa: E402


def load_installed_ops(artifact: str | None):
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return importlib.import_module("small_matrix_cholesky")
    finally:
        if artifact:
            sys.path.remove(artifact)


def make_spd(batch: int, n: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(41000 + n + batch)
    x = torch.randn(
        batch,
        n,
        n,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    ) / n**0.5
    return (
        x @ x.transpose(-1, -2)
        + 0.5 * torch.eye(n, device="cuda", dtype=torch.float32)
    ).contiguous()


def median_ms(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=["source", "installed"], default="source"
    )
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--registration-include", default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    ops = (
        load_source_ops(args.registration_include)
        if args.backend == "source"
        else load_installed_ops(args.artifact)
    )

    shapes = [(4096, 32), (1024, 64), (256, 128)]
    candidate_times: list[float] = []
    baseline_times: list[float] = []
    print("batch,n,candidate_ms,pytorch_ms,speedup,candidate_tflops,io_gbps")
    for batch, n in shapes:
        input = make_spd(batch, n)
        candidate_output = torch.empty_like(input)
        baseline_output = torch.empty_like(input)
        info = torch.empty(batch, device="cuda", dtype=torch.int32)

        def candidate() -> None:
            ops.cholesky_small_fp32(input, out=candidate_output)

        def baseline() -> None:
            torch.linalg.cholesky_ex(
                input,
                check_errors=False,
                out=(baseline_output, info),
            )

        candidate()
        baseline()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            candidate_output,
            baseline_output,
            rtol=5e-4,
            atol=2e-4,
        )

        candidate_ms = median_ms(candidate, args.warmup, args.iterations)
        baseline_ms = median_ms(baseline, args.warmup, args.iterations)
        flops = batch * n**3 / 3.0
        tflops = flops / (candidate_ms * 1e-3) / 1e12
        io_bytes = 2 * batch * n * n * 4
        io_gbps = io_bytes / (candidate_ms * 1e-3) / 1e9
        candidate_times.append(candidate_ms)
        baseline_times.append(baseline_ms)
        print(
            f"{batch},{n},{candidate_ms:.6f},{baseline_ms:.6f},"
            f"{baseline_ms / candidate_ms:.3f},{tflops:.3f},{io_gbps:.3f}"
        )

    candidate_geomean = math.exp(
        sum(math.log(value) for value in candidate_times)
        / len(candidate_times)
    )
    baseline_geomean = math.exp(
        sum(math.log(value) for value in baseline_times)
        / len(baseline_times)
    )
    print(f"candidate_geomean_ms={candidate_geomean:.6f}")
    print(f"pytorch_geomean_ms={baseline_geomean:.6f}")
    print(f"geomean_speedup={baseline_geomean / candidate_geomean:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
