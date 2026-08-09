#!/usr/bin/env python3
"""FP8 causal GQA prefill attention benchmark (kernel vs eager SDPA vs torch.compile)."""

from __future__ import annotations

import argparse
import importlib
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def _apply_mem_cap(max_mem_gb: float = 30.0) -> None:
    if not torch.cuda.is_available() or max_mem_gb <= 0:
        return
    total = torch.cuda.get_device_properties(0).total_memory
    cap = int(max_mem_gb * 1024**3)
    if total <= 0 or cap >= total:
        return
    torch.cuda.set_per_process_memory_fraction(cap / total)


def elapsed_us(fn, warmup: int = 20, repeats: int = 50) -> float:
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
        return importlib.import_module("fp8_prefill_attention_blackwell")
    if artifact:
        sys.path.insert(0, artifact)
    return importlib.import_module("fp8_prefill_attention_blackwell")


def run_case(ops, label: str, s: int) -> dict:
    q = (torch.randn(s, 32, 128, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    k = (torch.randn(s, 8, 128, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    v = (torch.randn(s, 8, 128, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    scale = 1.0 / math.sqrt(128)

    def kernel():
        return ops.fp8_causal_gqa_attention_bf16(q, k, v, softmax_scale=scale)

    qb = q.float().to(torch.bfloat16).transpose(0, 1)
    kb = k.float().to(torch.bfloat16).transpose(0, 1)
    vb = v.float().to(torch.bfloat16).transpose(0, 1)
    # GQA: expand KV heads 8 -> 32 to match Hq for SDPA.
    kb_e = kb.repeat_interleave(4, dim=0)
    vb_e = vb.repeat_interleave(4, dim=0)

    def eager():
        return F.scaled_dot_product_attention(qb, kb_e, vb_e, is_causal=True)

    eager_us = elapsed_us(eager)
    compiled = torch.compile(eager, mode="reduce-overhead")
    compiled()
    compile_us = elapsed_us(lambda: compiled())
    kernel_us = elapsed_us(kernel)
    return {
        "label": label, "S": s, "Hq": 32, "Hkv": 8, "D": 128,
        "kernel_us": kernel_us, "eager_us": eager_us, "compile_us": compile_us,
        "vs_eager": eager_us / kernel_us,
        "vs_compile": compile_us / kernel_us,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="installed")
    parser.add_argument("--artifact")
    parser.add_argument("--max-mem-gb", type=float, default=30.0)
    args = parser.parse_args()
    _apply_mem_cap(args.max_mem_gb)
    ops = load_ops(args.backend, args.artifact)
    print("label,S,Hq,Hkv,D,kernel_us,eager_us,compile_us,vs_eager,vs_compile")
    for label, s in [
        ("prefill_1k", 1024),
        ("prefill_2k", 2048),
        ("prefill_4k", 4096),
    ]:
        r = run_case(ops, label, s)
        print(
            f"{r['label']},{r['S']},{r['Hq']},{r['Hkv']},{r['D']},"
            f"{r['kernel_us']:.3f},{r['eager_us']:.3f},{r['compile_us']:.3f},"
            f"{r['vs_eager']:.2f}x,{r['vs_compile']:.2f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
