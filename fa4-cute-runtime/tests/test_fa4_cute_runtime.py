#!/usr/bin/env python3
"""Strict GROOT-shape tests for the forward-only FA4 CuTe package."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch


PACKAGE = Path(__file__).resolve().parents[1]


def load_ops(backend: str, artifact: str | None):
    root = Path(artifact) if artifact else PACKAGE / "torch-ext"
    sys.path.insert(0, str(root))
    try:
        return importlib.import_module("fa4_cute_runtime")
    finally:
        sys.path.remove(str(root))


def metrics(got: torch.Tensor, ref: torch.Tensor):
    diff = (got.float() - ref.float()).abs().flatten()
    cosine = torch.nn.functional.cosine_similarity(
        got.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    return float(diff.max()), float(torch.quantile(diff, 0.99)), float(diff.mean()), float(cosine)


def run_case(
    ops,
    *,
    sq: int,
    sk: int,
    hq: int,
    hk: int,
    d: int,
    causal: bool,
    valid_k: int | None = None,
):
    torch.manual_seed(2026 + sq + sk + d)
    q = torch.randn((1, sq, hq, d), device="cuda", dtype=torch.float16)
    k = torch.randn((1, sk, hk, d), device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    seqused_k = None
    if valid_k is not None:
        if not 0 < valid_k <= sk:
            raise ValueError(f"valid_k must be in [1, {sk}], got {valid_k}")
        seqused_k = torch.tensor([valid_k], device="cuda", dtype=torch.int32)

    got = torch.empty_like(q)
    ops.forward_static(
        q, k, v, got, causal=causal, seqused_k=seqused_k
    )
    ref_k = k[:, :valid_k] if valid_k is not None else k
    ref_v = v[:, :valid_k] if valid_k is not None else v
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2).float(), ref_k.transpose(1, 2).float(),
        ref_v.transpose(1, 2).float(), is_causal=causal,
        enable_gqa=hq != hk,
    ).transpose(1, 2).half()
    torch.cuda.synchronize()
    max_abs, p99_abs, mean_abs, cosine = metrics(got, ref)
    if p99_abs > 0.00390625 or cosine < 0.999:
        raise AssertionError(
            f"sq={sq} sk={sk} valid_k={valid_k} h={hq}/{hk} "
            f"d={d} causal={causal}: "
            f"max={max_abs} p99={p99_abs} mean={mean_abs} cos={cosine}"
        )

    static_out = torch.empty_like(q)
    ops.forward_static(
        q, k, v, static_out, causal=causal, seqused_k=seqused_k
    )
    torch.cuda.synchronize()
    static_metrics = metrics(static_out, got)
    if static_metrics[0] != 0.0:
        raise AssertionError(f"static entry differs from public entry: {static_metrics}")

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = ops.forward_static(
            q, k, v, static_out, causal=causal, seqused_k=seqused_k
        )
    graph.replay()
    first = graph_out.clone()
    graph.replay()
    torch.cuda.synchronize()
    if not torch.equal(first, graph_out):
        raise AssertionError("FA4 CUDA Graph replay is not bitwise deterministic")
    print(
        f"PASS sq={sq} sk={sk} valid_k={valid_k} h={hq}/{hk} "
        f"d={d} causal={causal} "
        f"max={max_abs:.6f} p99={p99_abs:.6f} mean={mean_abs:.6f} "
        f"cos={cosine:.8f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (11, 0):
        raise SystemExit("This release gate requires SM110")
    ops = load_ops(args.backend, args.artifact)
    cases = [dict(sq=41, sk=41, hq=32, hk=32, d=48, causal=False)]
    if args.mode == "full":
        cases.extend([
            dict(sq=277, sk=277, hq=16, hk=16, d=72, causal=False),
            dict(sq=277, sk=277, hq=16, hk=8, d=128, causal=True),
            dict(sq=1024, sk=1024, hq=16, hk=8, d=128, causal=True),
            # PI0.5 Thor encoder profiles. The fixed-shape runtime pads K/V
            # and supplies the real length through seqused_k.
            dict(sq=320, sk=320, hq=8, hk=1, d=256, causal=False),
            dict(sq=456, sk=968, valid_k=456, hq=8, hk=1, d=256, causal=False),
            dict(sq=712, sk=968, valid_k=712, hq=8, hk=1, d=256, causal=False),
            dict(sq=968, sk=968, hq=8, hk=1, d=256, causal=False),
        ])
    for case in cases:
        run_case(ops, **case)
    print(f"fa4-cute-runtime {args.backend} {args.mode}: passed {len(cases)}/{len(cases)}")


if __name__ == "__main__":
    main()
