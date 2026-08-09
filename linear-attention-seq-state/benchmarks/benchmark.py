#!/usr/bin/env python3
"""Gated Delta recurrent seq-state benchmark (kernel vs reference vs torch.compile)."""

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


def elapsed_us(fn, warmup: int = 10, repeats: int = 50) -> float:
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
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "torch-ext"))
        return importlib.import_module("linear_attention_seq_state")
    if artifact:
        sys.path.insert(0, artifact)
    return importlib.import_module("linear_attention_seq_state")


def reference_seq(q, k, v, g, beta, state0):
    qf, kf, vf = q.float(), k.float(), v.float()
    gf, bf = g.float(), beta.float()
    st = state0.float().clone()
    out = torch.empty_like(qf)
    inv_sqrt = 1.0 / (q.shape[-1] ** 0.5)
    for s in range(q.shape[0]):
        qs = qf[s].clone() * inv_sqrt
        for h in range(q.shape[1]):
            st[h] = st[h] * torch.exp(gf[s, h])
            kv_mem = torch.mv(st[h].t(), kf[s, h])
            delta = (vf[s, h] - kv_mem) * bf[s, h]
            st[h] = st[h] + torch.outer(kf[s, h], delta)
            out[s, h] = torch.mv(st[h].t(), qs[h])
    return out.to(torch.bfloat16), st.to(torch.bfloat16)


def run_case(ops, label: str, s: int, h: int, d: int = 128, do_compile: bool = True) -> dict:
    gen = torch.Generator(device="cuda").manual_seed(0)
    q = (torch.randn((s, h, d), device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
    k = (torch.randn((s, h, d), device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
    v = (torch.randn((s, h, d), device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
    g = (torch.randn((s, h), device="cuda", generator=gen) * 0.01).to(torch.bfloat16)
    beta = torch.sigmoid(torch.randn((s, h), device="cuda", generator=gen)).to(torch.bfloat16)
    state0 = (torch.randn((h, d, d), device="cuda", generator=gen) * 0.01).to(torch.bfloat16)
    state = state0.clone()
    out = torch.empty_like(q)

    def kernel():
        ops.gated_delta_recurrent_seq_bf16(q, k, v, g, beta, state, out=out)
        return out, state

    kernel_us = elapsed_us(kernel)

    ref = lambda: reference_seq(q, k, v, g, beta, state0)
    ref_us = elapsed_us(ref)

    if do_compile:
        compiled = torch.compile(ref, mode="reduce-overhead")
        compiled()
        compile_us = elapsed_us(lambda: compiled())
    else:
        compile_us = float("nan")

    return {
        "label": label, "S": s, "H": h, "D": d,
        "kernel_us": kernel_us, "reference_us": ref_us, "compile_us": compile_us,
        "vs_reference": ref_us / kernel_us,
        "vs_compile": compile_us / kernel_us,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="installed")
    parser.add_argument("--artifact")
    parser.add_argument("--max-mem-gb", type=float, default=30.0)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="run the torch.compile reference region (pathologically slow on SM110; "
        "skipped by default there)",
    )
    args = parser.parse_args()
    _apply_mem_cap(args.max_mem_gb)
    ops = load_ops(args.backend, args.artifact)
    do_compile = args.compile or torch.cuda.get_device_capability(0)[0] != 11
    print("label,S,H,D,kernel_us,reference_us,compile_us,vs_reference,vs_compile")
    for label, s, h in [
        ("seq_1k", 1024, 2),
        ("seq_2k", 2048, 4),
        ("seq_4k", 4096, 4),
    ]:
        r = run_case(ops, label, s, h, do_compile=do_compile)
        print(
            f"{r['label']},{r['S']},{r['H']},{r['D']},{r['kernel_us']:.3f},"
            f"{r['reference_us']:.3f},{r['compile_us']:.3f},"
            f"{r['vs_reference']:.2f}x,{r['vs_compile']:.2f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
