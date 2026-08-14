#!/usr/bin/env python3
"""Benchmark static SageAttention3 against PyTorch SDPA."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "sageattention3-blackwell" / "tests" / "test_sageattention3_blackwell.py"
SAGE2_TEST = ROOT / "sageattention2-blackwell" / "tests" / "test_sageattention2_blackwell.py"


def load_file_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def time_cuda(fn, warmup: int, iters: int) -> float:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("source", "installed"), default="source")
    parser.add_argument("--artifact")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    test = load_file_module("sage3_test", TEST)
    sage2_test = load_file_module("sage2_test", SAGE2_TEST)
    if args.backend == "source":
        ops = test.load_source_ops()
        sage2 = sage2_test.load_source_ops()
    else:
        from kernels import get_kernel

        ops = test.InstalledOps(test.load_installed_module(args.artifact))
        sage2 = get_kernel(
            "flashrt/sageattention2-blackwell",
            version=1,
            trust_remote_code=True,
        )

    cases = [(6144, 128), (2688, 64)]
    if args.mode == "full":
        cases = [(6144, 128), (24576, 128), (2688, 64)]
    print("| S | D | SDPA us | Sage2 static us | Sage3 core+quant us | Sage3 fused eager us | Sage3 fused graph us | graph vs SDPA | fused/legacy cosine |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s, d in cases:
        q = torch.randn((1, s, 32, d), device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        qn, kn, vn, delta_s, qh, kh, vh = test.preprocess(q, k, v, False)
        ws = list(test.alloc(qn))
        fused_ws = test.alloc_fused(q)
        out = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh)
        sage2_ws = (
            sage2.allocate_workspace(qn, kn, vn, fp8v=True) if d == 128 else None
        )
        sage2_out = torch.empty_like(qn)

        def run_sdpa():
            return torch.nn.functional.scaled_dot_product_attention(qh, kh, vh)

        def run_sage3():
            ops.quantize_q_fp4_nhd(qn, ws[0], ws[3])
            ops.quantize_k_fp4_nhd(kn, ws[1], ws[4])
            ops.quantize_v_fp4_nhd(vn, ws[2], ws[5])
            return ops.attention(ws, delta_s, s, False)

        def run_sage3_fused():
            return ops.fused(q, k, v, fused_ws)

        def run_sage2():
            return sage2.sage2_prefill_fp8v_bf16_d128(
                qn, kn, vn, out=sage2_out, workspace=sage2_ws
            )

        got = run_sage3()
        fused_got = run_sage3_fused()
        qnb, knb, vnb, dsb, *_ = test.preprocess(q, k, v, True)
        legacy_block_ws = list(test.alloc(qnb))
        ops.quantize_q_fp4_nhd(qnb, legacy_block_ws[0], legacy_block_ws[3])
        ops.quantize_k_fp4_nhd(knb, legacy_block_ws[1], legacy_block_ws[4])
        ops.quantize_v_fp4_nhd(vnb, legacy_block_ws[2], legacy_block_ws[5])
        legacy_block = ops.attention(legacy_block_ws, dsb, s, True)[:, :s]

        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            graph_output = run_sage3_fused()

        def replay_sage3_fused():
            graph.replay()
            return graph_output

        sdpa_us = time_cuda(run_sdpa, args.warmup, args.iters)
        sage2_us = (
            time_cuda(run_sage2, args.warmup, args.iters) if sage2_ws else float("nan")
        )
        sage_us = time_cuda(run_sage3, args.warmup, args.iters)
        fused_us = time_cuda(run_sage3_fused, args.warmup, args.iters)
        fused_graph_us = time_cuda(replay_sage3_fused, args.warmup, args.iters)
        print(
            f"| {s} | {d} | {sdpa_us:.3f} | {sage2_us:.3f} | {sage_us:.3f} | "
            f"{fused_us:.3f} | {fused_graph_us:.3f} | "
            f"{sdpa_us / fused_graph_us:.2f}x | "
            f"{test.cosine(fused_got, legacy_block):.8f} |"
        )


if __name__ == "__main__":
    main()
