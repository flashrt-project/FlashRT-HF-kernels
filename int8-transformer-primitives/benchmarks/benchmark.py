#!/usr/bin/env python3
"""Benchmark INT8 transformer primitives."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "int8-transformer-primitives" / "tests"))
from test_int8_transformer_primitives import load_source_ops  # noqa: E402


def load_ops(backend: str, artifact: str | None):
    if backend == "source":
        return load_source_ops()
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return importlib.import_module("int8_transformer_primitives")
    finally:
        if artifact:
            sys.path.remove(artifact)


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--mode", choices=["headline", "full"], default="headline")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    ops = load_ops(args.backend, args.artifact)

    shapes = [
        ("decode_m8", 8, 1024, 2560),
        ("small_batch", 64, 2048, 8192),
        ("vision_prefill", 522, 2048, 2560),
    ]
    if args.mode == "full":
        shapes += [
            ("m1", 1, 1024, 1024),
            ("m17", 17, 256, 256),
            ("wide_ffn", 257, 2048, 8192),
        ]

    print("workload,M,K,N,op,flashrt_us,torch_eager_us,speedup")
    for name, m, k, n in shapes:
        x = (torch.randn((m, k), device="cuda") * 0.5).to(torch.bfloat16)
        w = (torch.randn((n, k), device="cuda") * 0.5).to(torch.bfloat16)
        x_i8, x_scale = ops.quantize_int8_rowwise_bf16(x)
        w_i8, w_scale = ops.quantize_int8_rowwise_bf16(w)
        torch.cuda.synchronize()
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

        def flash():
            ops.int8_rowwise_linear_bf16(x_i8, w_i8, x_scale, w_scale, out=out)

        def eager():
            ((x_i8.float() @ w_i8.float().t()) * x_scale[:, None] * w_scale[None, :]).to(torch.bfloat16)

        fu = time_us(flash, args.warmup, args.iters)
        eu = time_us(eager, max(5, args.warmup // 2), max(20, args.iters // 2))
        print(f"{name},{m},{k},{n},int8_rowwise_linear_bf16,{fu:.3f},{eu:.3f},{eu/fu:.2f}x")

    q = (torch.randn((522, 2048), device="cuda") * 0.5).to(torch.bfloat16)
    weight = torch.randn((2048,), device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(q, dtype=torch.int8)
    scales = torch.empty((q.shape[0],), device="cuda", dtype=torch.float32)

    def flash_rms():
        ops.rms_norm_quantize_int8_rowwise_bf16(q, weight, out=out, scales=scales)

    def eager_rms():
        y = q.float() * torch.rsqrt((q.float() * q.float()).mean(dim=1, keepdim=True) + 1e-6) * weight.float()
        s = torch.clamp(y.abs().amax(dim=1) / 127.0, min=1e-10)
        torch.clamp(torch.round(y / s[:, None]), -127, 127).to(torch.int8)

    fu = time_us(flash_rms, args.warmup, args.iters)
    eu = time_us(eager_rms, max(5, args.warmup // 2), max(20, args.iters // 2))
    print(f"vision_prefill,522,2048,0,rms_norm_quantize_int8_rowwise_bf16,{fu:.3f},{eu:.3f},{eu/fu:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
