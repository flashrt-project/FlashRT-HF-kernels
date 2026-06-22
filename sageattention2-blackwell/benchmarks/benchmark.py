#!/usr/bin/env python3
"""Benchmark sageattention2-blackwell against PyTorch SDPA."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sageattention2-blackwell" / "tests"))
from test_sageattention2_blackwell import load_source_ops, make_inputs, reference, stats  # noqa: E402


def load_installed_ops(artifact: str | None):
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return importlib.import_module("sageattention2_blackwell")
    finally:
        if artifact:
            sys.path.remove(artifact)


def time_cuda(fn, iters: int, warmup: int) -> float:
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


def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    q_t = q.transpose(1, 2)
    if q.shape[2] != k.shape[2]:
        repeat = q.shape[2] // k.shape[2]
        k = k.repeat_interleave(repeat, dim=2)
        v = v.repeat_interleave(repeat, dim=2)
    return F.scaled_dot_product_attention(q_t, k.transpose(1, 2), v.transpose(1, 2), is_causal=causal).transpose(1, 2)


def run_case(ops, name: str, seqlen: int, q_heads: int, kv_heads: int, causal: bool, iters: int, warmup: int):
    q, k, v = make_inputs(1, seqlen, q_heads, kv_heads)
    out = torch.empty_like(q, dtype=torch.bfloat16)
    ref = reference(q, k, v, causal)

    q_i8, q_scale = ops.quantize_q_bf16_d128(q)
    k_i8, k_scale = ops.quantize_k_bf16_d128(k)
    v_half = ops.quantize_v_fp16_bf16_d128(v)
    torch.cuda.synchronize()

    def run_sdpa():
        return sdpa(q, k, v, causal)

    def run_core():
        return ops.sage2_qk_int8_sv_f16_bf16_d128(
            q_i8, k_i8, v_half, q_scale, k_scale, causal=causal, out=out
        )

    def run_bf16():
        return ops.sage2_prefill_f16_bf16_d128(q, k, v, causal=causal, out=out)

    got = run_core()
    torch.cuda.synchronize()
    s = stats(got, ref)
    sdpa_us = time_cuda(run_sdpa, iters, warmup)
    core_us = time_cuda(run_core, iters, warmup)
    bf16_us = time_cuda(run_bf16, iters, warmup)
    return {
        "name": name,
        "seqlen": seqlen,
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "causal": causal,
        "sdpa_us": sdpa_us,
        "core_us": core_us,
        "bf16_us": bf16_us,
        "core_speedup": sdpa_us / core_us,
        "bf16_speedup": sdpa_us / bf16_us,
        **s,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--markdown", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    major, _minor = torch.cuda.get_device_capability(0)
    if major < 12:
        raise SystemExit("sageattention2-blackwell requires Blackwell-class CUDA capability")
    torch.manual_seed(2026)
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    cases = [
        ("qwen3_prefill", 1024, 32, 8, True),
        ("wan_self_attn", 1024, 24, 24, False),
    ]
    if args.mode == "full":
        cases = [
            ("qwen3_prefill", 1024, 32, 8, True),
            ("qwen3_prefill", 2048, 32, 8, True),
            ("qwen3_prefill", 4096, 32, 8, True),
            ("qwen3_prefill", 8192, 32, 8, True),
            ("wan_self_attn", 1024, 24, 24, False),
            ("wan_self_attn", 2520, 24, 24, False),
            ("wan_self_attn", 4096, 24, 24, False),
            ("wan_self_attn", 5070, 24, 24, False),
        ]
    rows = [run_case(ops, *case, args.iters, args.warmup) for case in cases]
    lines = [
        "| Workload | S | Hq/Hkv | Mask | SDPA us | Sage core us | Core speedup | BF16 wrapper us | Wrapper speedup | Cos | p99 abs |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        line = (
            f"| {row['name']} | {row['seqlen']} | {row['q_heads']}/{row['kv_heads']} | "
            f"{'causal' if row['causal'] else 'none'} | {row['sdpa_us']:.3f} | {row['core_us']:.3f} | "
            f"{row['core_speedup']:.2f}x | {row['bf16_us']:.3f} | {row['bf16_speedup']:.2f}x | "
            f"{row['cos']:.6f} | {row['p99_abs']:.6f} |"
        )
        print(line)
        lines.append(line)
    if args.markdown:
        Path(args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
