#!/usr/bin/env python3
"""Benchmark causal-conv1d-state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
import torch

TESTS = Path(__file__).resolve().parents[1] / "tests"
sys.path.insert(0, str(TESTS))
from test_causal_conv1d_state import MODES, SHAPES, load_installed_ops, load_source_ops, make_inputs, ref_chunk  # noqa: E402


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
    return float(start.elapsed_time(end) * 1000.0 / iters)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--mode", choices=sorted(MODES), default="headline")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    rows = []
    for name in MODES[args.mode]:
        kind, B, S, C, K = SHAPES[name]
        x, w, bias, state = make_inputs(B, S, C, K, seed=9900 + C + S)
        if kind in {"chunk", "parallel", "gqa"}:
            if kind == "gqa":
                def kernel_call():
                    state_work = state.clone()
                    return ops.gqa(x, w, state_work, bias)
            elif kind == "parallel":
                def kernel_call():
                    state_work = state.clone()
                    return ops.parallel(x, w, state_work, bias)
            else:
                def kernel_call():
                    state_work = state.clone()
                    return ops.chunk(x, w, state_work, bias)

            def ref_call():
                return ref_chunk(x, w, bias, state)
        else:
            continue
        kernel_us = time_cuda(kernel_call, args.warmup, args.iters)
        ref_us = time_cuda(ref_call, max(2, args.warmup // 5), max(5, args.iters // 20))
        rows.append({"shape": name, "kind": kind, "B": B, "S": S, "C": C, "K": K, "kernel_us": kernel_us, "torch_reference_us": ref_us, "speedup": ref_us / kernel_us})
        print(f"{name}: kernel={kernel_us:.3f}us ref={ref_us:.3f}us speedup={ref_us / kernel_us:.2f}x")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
