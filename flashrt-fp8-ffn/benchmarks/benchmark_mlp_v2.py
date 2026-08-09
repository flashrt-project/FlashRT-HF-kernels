#!/usr/bin/env python3
"""Compare staged-bias v1 and fused-bias v2 FP8 GELU MLP entries."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
TEST_FILE = ROOT / "flashrt-fp8-ffn" / "tests" / "test_fp8_ffn.py"
SHAPES = {
    "groot_vit": (128, 1024, 4096, 1024),
    "groot_deepstack": (128, 4096, 4096, 2048),
    "groot_action_dit": (41, 1536, 6144, 1536),
    "pi05_decoder": (10, 1024, 4096, 1024),
}


def helpers_module():
    spec = importlib.util.spec_from_file_location("fp8_ffn_helpers", TEST_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {TEST_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def time_us(fn, warmup: int, iterations: int, rounds: int = 5) -> float:
    samples = []
    for _ in range(rounds):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iterations)
    return float(statistics.median(samples))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("source", "installed"), default="source")
    parser.add_argument("--artifact")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--json-out")
    args = parser.parse_args()
    helpers = helpers_module()
    if args.backend == "source":
        ops = helpers.load_source_ops()
    else:
        if args.artifact:
            sys.path.insert(0, args.artifact)
        try:
            ops = importlib.import_module("flashrt_fp8_ffn")
        finally:
            if args.artifact:
                sys.path.remove(args.artifact)

    rows = []
    for name, shape in SHAPES.items():
        tensors = helpers.make_case(*shape)
        m, _, h, n = shape
        hidden_v1 = torch.empty((m, h), device="cuda", dtype=torch.bfloat16)
        hidden_fp8_v1 = torch.empty_like(hidden_v1, dtype=helpers.fp8_dtype())
        out_v1_buf = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        hidden_v2 = torch.empty_like(hidden_v1)
        hidden_fp8_v2 = torch.empty_like(hidden_fp8_v1)
        out_v2_buf = torch.empty_like(out_v1_buf)
        if args.backend == "source":
            v1 = lambda: ops.fp8_gelu_mlp_bf16(
                *tensors, hidden=hidden_v1, hidden_fp8=hidden_fp8_v1,
                out=out_v1_buf,
            )
            v2 = lambda: ops.fp8_gelu_mlp_v2_bf16(
                *tensors, hidden=hidden_v2, hidden_fp8=hidden_fp8_v2,
                out=out_v2_buf,
            )
        else:
            v1 = lambda: ops.fp8_gelu_mlp_bf16(
                *tensors, hidden_bf16=hidden_v1, hidden_fp8=hidden_fp8_v1,
                out=out_v1_buf,
            )
            v2 = lambda: ops.fp8_gelu_mlp_v2_bf16(
                *tensors, hidden_bf16=hidden_v2, hidden_fp8=hidden_fp8_v2,
                out=out_v2_buf,
            )
        out_v1 = v1()
        out_v2 = v2()
        torch.cuda.synchronize()
        old = os.environ.get("FLASHRT_FP8_FFN_REQUIRE_BIAS_EPILOGUE")
        os.environ["FLASHRT_FP8_FFN_REQUIRE_BIAS_EPILOGUE"] = "1"
        try:
            v2()
            torch.cuda.synchronize()
            fused_hit = True
        except RuntimeError as error:
            if "BIAS epilogue was required" not in str(error):
                raise
            fused_hit = False
        finally:
            if old is None:
                os.environ.pop("FLASHRT_FP8_FFN_REQUIRE_BIAS_EPILOGUE", None)
            else:
                os.environ["FLASHRT_FP8_FFN_REQUIRE_BIAS_EPILOGUE"] = old
        metrics = helpers.distribution_metrics(out_v2, out_v1)
        v1_us = time_us(v1, args.warmup, args.iterations)
        v2_us = time_us(v2, args.warmup, args.iterations)
        row = {
            "shape": name,
            "M": shape[0],
            "K": shape[1],
            "H": shape[2],
            "N": shape[3],
            "fused_bias_epilogue": fused_hit,
            "v1_us": v1_us,
            "v2_us": v2_us,
            "speedup": v1_us / v2_us,
            **metrics,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    result = {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "rows": rows,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
