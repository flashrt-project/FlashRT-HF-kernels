#!/usr/bin/env python3
"""RTX benchmark for grouped W4A4 expert projection launch aggregation."""

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


PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "tests"))
from test_grouped_moe_gemv import load_source_ops, sfb_bytes  # noqa: E402


def elapsed_us(fn, warmup: int = 20, repeats: int = 100) -> float:
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
        return load_source_ops()
    if artifact:
        sys.path.insert(0, artifact)
    return importlib.import_module("grouped_moe_gemv")


def run_case(ops, label: str, m: int, top_k: int, n: int, k: int) -> dict:
    experts = 8
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 0.2
    packed = torch.empty((m, k // 2), device="cuda", dtype=torch.uint8)
    sfa = torch.empty((sfb_bytes(m, k),), device="cuda", dtype=torch.uint8)
    weights = torch.full(
        (experts, n, k // 2), 0x11, device="cuda", dtype=torch.uint8
    )
    sfb = torch.full(
        (experts, sfb_bytes(n, k)), 0x38, device="cuda", dtype=torch.uint8
    )
    alpha = torch.ones((experts,), device="cuda", dtype=torch.float32)
    idx = (
        torch.arange(m * top_k, device="cuda", dtype=torch.int32)
        .reshape(m, top_k)
        .remainder(experts)
        .contiguous()
    )
    out = torch.empty((m, top_k, n), device="cuda", dtype=torch.bfloat16)
    routed_x = x[:, None, :].expand(m, top_k, k).reshape(m * top_k, k).contiguous()
    w4a16_out = torch.empty((m * top_k, n), device="cuda", dtype=torch.bfloat16)

    def grouped_region():
        ops.quantize_activations_nvfp4_bf16(x, packed=packed, sfa=sfa)
        ops.grouped_w4a4_gemv_bf16(
            packed, weights, sfa, sfb, alpha, idx, out=out
        )

    def grouped_kernel():
        ops.grouped_w4a4_gemv_bf16(
            packed, weights, sfa, sfb, alpha, idx, out=out
        )

    route_out = torch.empty((m, 1, n), device="cuda", dtype=torch.bfloat16)
    route_idx = [idx[:, route : route + 1].contiguous() for route in range(top_k)]

    def route_loop():
        ops.quantize_activations_nvfp4_bf16(x, packed=packed, sfa=sfa)
        for route in range(top_k):
            ops.grouped_w4a4_gemv_bf16(
                packed,
                weights,
                sfa,
                sfb,
                alpha,
                route_idx[route],
                out=route_out,
            )

    pair_packed = torch.empty((1, k // 2), device="cuda", dtype=torch.uint8)
    pair_sfa = torch.empty((sfb_bytes(1, k),), device="cuda", dtype=torch.uint8)
    pair_out = torch.empty((1, 1, n), device="cuda", dtype=torch.bfloat16)
    pair_idx = [
        idx[token : token + 1, route : route + 1].contiguous()
        for token in range(m)
        for route in range(top_k)
    ]

    def pair_loop():
        pair = 0
        for token in range(m):
            for _route in range(top_k):
                ops.quantize_activations_nvfp4_bf16(
                    x[token : token + 1], packed=pair_packed, sfa=pair_sfa
                )
                ops.grouped_w4a4_gemv_bf16(
                    pair_packed,
                    weights,
                    pair_sfa,
                    sfb,
                    alpha,
                    pair_idx[pair],
                    out=pair_out,
                )
                pair += 1

    grouped_us = elapsed_us(grouped_region)
    grouped_kernel_us = elapsed_us(grouped_kernel)
    def w4a16_region():
        ops.grouped_w4a16_gemv_bf16(
            routed_x,
            weights,
            sfb,
            alpha,
            idx.reshape(-1),
            w_stride=n * k // 2,
            sfb_stride=sfb.shape[1],
            n=n,
            out=w4a16_out,
        )

    w4a16_us = elapsed_us(w4a16_region)
    route_us = elapsed_us(route_loop) if top_k > 1 else grouped_us
    pair_repeats = 30 if m * top_k >= 32 else 100
    pair_us = elapsed_us(pair_loop, warmup=5, repeats=pair_repeats)
    return {
        "label": label,
        "M": m,
        "top_k": top_k,
        "pairs": m * top_k,
        "N": n,
        "K": k,
        "grouped_us": grouped_us,
        "grouped_kernel_us": grouped_kernel_us,
        "route_loop_us": route_us,
        "pair_loop_us": pair_us,
        "w4a16_us": w4a16_us,
        "vs_route_loop": route_us / grouped_us,
        "vs_pair_loop": pair_us / grouped_us,
        "vs_w4a16": w4a16_us / grouped_us,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact")
    parser.add_argument("--max-mem-gb", type=float, default=30.0)
    args = parser.parse_args()
    _apply_mem_cap(args.max_mem_gb)
    ops = load_ops(args.backend, args.artifact)
    cases = [
        ("gate_up_decode", 1, 8, 1024, 2048),
        ("gate_up_verify", 7, 8, 1024, 2048),
        ("down_decode", 8, 1, 2048, 512),
        ("down_verify", 56, 1, 2048, 512),
    ]
    print("label,M,top_k,pairs,N,K,w4a4_region_us,w4a4_kernel_us,w4a16_us,route_loop_us,pair_loop_us,vs_w4a16,vs_route,vs_pair")
    for case in cases:
        result = run_case(ops, *case)
        print(
            f"{result['label']},{result['M']},{result['top_k']},{result['pairs']},"
            f"{result['N']},{result['K']},{result['grouped_us']:.3f},"
            f"{result['grouped_kernel_us']:.3f},{result['w4a16_us']:.3f},"
            f"{result['route_loop_us']:.3f},"
            f"{result['pair_loop_us']:.3f},{result['vs_w4a16']:.2f}x,"
            f"{result['vs_route_loop']:.2f}x,{result['vs_pair_loop']:.2f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
