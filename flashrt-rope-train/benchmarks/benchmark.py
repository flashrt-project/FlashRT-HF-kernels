#!/usr/bin/env python3
"""RoPE training smoke test (reference-only package, no CUDA kernel yet).

Not a benchmark: runs each shape once and checks the op produces
same-shaped, finite outputs. Exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch


def load_ops(backend: str, artifact: str | None):
    if backend == "source":
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "torch-ext"))
        return importlib.import_module("flashrt_rope_train")
    if artifact:
        sys.path.insert(0, artifact)
    return importlib.import_module("flashrt_rope_train")


def apply_max_mem_cap(max_mem_gb: float) -> None:
    if not torch.cuda.is_available() or max_mem_gb <= 0:
        return
    total = torch.cuda.get_device_properties(0).total_memory
    cap = int(max_mem_gb * 1024**3)
    if total <= 0 or cap >= total:
        return
    torch.cuda.set_per_process_memory_fraction(cap / total)


def run_case(ops, label: str, b: int, h: int, t: int, d: int) -> None:
    torch.manual_seed(0)
    with torch.no_grad():
        q = torch.randn(b, h, t, d, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(b, h, t, d, device="cuda", dtype=torch.bfloat16)
        cos = torch.randn(b, t, d, device="cuda", dtype=torch.bfloat16)
        sin = torch.randn(b, t, d, device="cuda", dtype=torch.bfloat16)

        q_out, k_out = ops.apply_rope_train(q, k, cos, sin)
    del q, k, cos, sin

    assert q_out.shape == (b, h, t, d) and q_out.dtype == torch.bfloat16, (
        f"{label}: bad q_out {tuple(q_out.shape)}/{q_out.dtype}"
    )
    assert k_out.shape == (b, h, t, d) and k_out.dtype == torch.bfloat16, (
        f"{label}: bad k_out {tuple(k_out.shape)}/{k_out.dtype}"
    )
    assert torch.isfinite(q_out).all() and torch.isfinite(k_out).all(), (
        f"{label}: non-finite output"
    )
    print(f"ok {label} B={b} H={h} T={t} D={d} -> {tuple(q_out.shape)} {q_out.dtype}")

    del q_out, k_out
    torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="installed")
    parser.add_argument("--artifact")
    parser.add_argument("--max-mem-gb", type=float, default=30.0)
    args = parser.parse_args()
    apply_max_mem_cap(args.max_mem_gb)
    ops = load_ops(args.backend, args.artifact)
    for label, b, h, t, d in [
        ("rope_2k", 1, 32, 2048, 128),
        ("rope_4k", 1, 32, 4096, 128),
    ]:
        run_case(ops, label, b, h, t, d)
    print("all smoke cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
