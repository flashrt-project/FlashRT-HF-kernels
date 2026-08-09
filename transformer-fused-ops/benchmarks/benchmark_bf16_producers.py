#!/usr/bin/env python3
"""Benchmark the SM110 BF16-to-FP8 producer family."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "transformer-fused-ops" / "tests" / "test_transformer_fused_ops.py"


def load_ops(backend: str, artifact: str | None):
    spec = importlib.util.spec_from_file_location("transformer_fused_ops_test", TEST)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.load_source_ops() if backend == "source" else module.load_installed_ops(artifact)


def measure(fn, warmup: int, iterations: int, rounds: int = 7) -> float:
    samples = []
    for _ in range(rounds):
        for _ in range(warmup):
            fn()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iterations)
    return float(statistics.median(samples))


def accuracy(got: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    diff = (got.float() - expected.float()).abs().flatten()
    rank = max(1, (99 * diff.numel() + 99) // 100)
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "p99_abs": float(diff.kthvalue(rank).values.item()),
        "cosine": float(torch.nn.functional.cosine_similarity(
            got.float().flatten(), expected.float().flatten(), dim=0
        ).item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (11, 0):
        raise SystemExit("SM110 is required")
    ops = load_ops(args.backend, args.artifact)
    scale = torch.tensor([0.025], device="cuda", dtype=torch.float32)
    rows = []

    x = torch.randn((768, 4304), device="cuda", dtype=torch.bfloat16)
    quant_out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    quant = lambda: ops.quantize_fp8_static_bf16(x, scale, out=quant_out)
    quant_ref = lambda: x.float().div(scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    quant()
    rows.append({
        "op": "quantize_fp8_static_bf16", "shape": [768, 4304],
        "kernel_us": measure(quant, args.warmup, args.iterations),
        "eager_us": measure(quant_ref, args.warmup, args.iterations),
        **accuracy(quant_out, quant_ref()),
    })

    x = torch.randn((512, 1152), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((1152,), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn_like(weight)
    ln_out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    ln = lambda: ops.layer_norm_quant_fp8_static_bf16(
        x, weight, bias, scale, out=ln_out
    )
    ln_ref = lambda: torch.nn.functional.layer_norm(
        x.float(), (1152,), weight.float(), bias.float(), 1e-6
    ).to(torch.bfloat16).float().div(scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    ln()
    rows.append({
        "op": "layer_norm_quant_fp8_static_bf16", "shape": [512, 1152],
        "kernel_us": measure(ln, args.warmup, args.iterations),
        "eager_us": measure(ln_ref, args.warmup, args.iterations),
        **accuracy(ln_out, ln_ref()),
    })

    merged = (torch.randn((768, 2 * 3456), device="cuda") * 0.25).to(torch.bfloat16)
    geglu_out = torch.empty((768, 3456), device="cuda", dtype=torch.float8_e4m3fn)
    geglu = lambda: ops.gate_geglu_merged_quant_fp8_static_bf16(
        merged, scale, out=geglu_out
    )

    def geglu_ref():
        gate, up = merged.float().chunk(2, dim=-1)
        value = gate / (
            1.0 + torch.exp(-1.5957691216057308 * gate * (1.0 + 0.044715 * gate.square()))
        )
        return value.mul(up).div(scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

    geglu()
    rows.append({
        "op": "gate_geglu_merged_quant_fp8_static_bf16", "shape": [768, 6912],
        "kernel_us": measure(geglu, args.warmup, args.iterations),
        "eager_us": measure(geglu_ref, args.warmup, args.iterations),
        **accuracy(geglu_out, geglu_ref()),
    })

    for row in rows:
        row["speedup_vs_eager"] = row["eager_us"] / row["kernel_us"]
    payload = {"device": torch.cuda.get_device_name(), "rows": rows}
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n")
    return 0 if all(row["p99_abs"] == 0.0 and row["cosine"] >= 0.9999 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
