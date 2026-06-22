#!/usr/bin/env python3
"""Benchmark adaptive-layernorm-producers against eager producer chains."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "adaptive-layernorm-producers" / "tests"))
from test_adaptive_layernorm_producers import (  # noqa: E402
    load_source_ops,
    make_case,
    quant_fp8,
    ref_adaln,
    ref_layer_norm_no_affine,
)


def load_installed_ops(artifact: str | None):
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return importlib.import_module("adaptive_layernorm_producers")
    finally:
        if artifact:
            sys.path.remove(artifact)


def time_cuda(fn, iters: int = 200, warmup: int = 50) -> float:
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


def run_case(ops, name: str, rows: int, dim: int, eps: float, iters: int) -> dict[str, float | str | int]:
    x, scale, shift, _inv_s, act_scale, _scale_fp8, _shift_fp8, _scale_deq, _shift_deq = make_case(rows, dim)
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)

    def fused():
        ops.ada_layer_norm_quant_fp8_bf16(x, scale, shift, act_scale, eps, out=out)

    def eager():
        quant_fp8(ref_adaln(x, scale, shift, eps), act_scale)

    no_affine_out = torch.empty_like(x, dtype=torch.float8_e4m3fn)

    def fused_no_affine():
        ops.layer_norm_no_affine_quant_fp8_static_bf16(x, act_scale, eps, out=no_affine_out)

    def eager_no_affine():
        quant_fp8(ref_layer_norm_no_affine(x, eps), act_scale)

    fused_us = time_cuda(fused, iters=iters)
    eager_us = time_cuda(eager, iters=iters)
    fused_no_affine_us = time_cuda(fused_no_affine, iters=iters)
    eager_no_affine_us = time_cuda(eager_no_affine, iters=iters)
    return {
        "shape": name,
        "rows": rows,
        "dim": dim,
        "ada_fp8_us": fused_us,
        "ada_eager_us": eager_us,
        "ada_speedup": eager_us / fused_us,
        "no_affine_fp8_us": fused_no_affine_us,
        "no_affine_eager_us": eager_no_affine_us,
        "no_affine_speedup": eager_no_affine_us / fused_no_affine_us,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--markdown", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(2026)
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    shapes = [
        ("decode_action", 16, 2048),
        ("wan_video_short", 64, 3072),
        ("wan_video_ctx", 256, 3072),
        ("wan_video_2k", 2520, 3072),
        ("wan_video_4k", 4096, 3072),
    ]
    rows = [run_case(ops, name, r, d, 1e-5, args.iters) for name, r, d in shapes]
    lines = [
        "| Shape | Rows | Dim | AdaLN->FP8 us | Eager chain us | Speedup | LN->FP8 us | Eager LN chain us | Speedup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        line = (
            f"| {row['shape']} | {row['rows']} | {row['dim']} | "
            f"{row['ada_fp8_us']:.3f} | {row['ada_eager_us']:.3f} | {row['ada_speedup']:.2f}x | "
            f"{row['no_affine_fp8_us']:.3f} | {row['no_affine_eager_us']:.3f} | {row['no_affine_speedup']:.2f}x |"
        )
        lines.append(line)
        print(line)
    if args.markdown:
        Path(args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
