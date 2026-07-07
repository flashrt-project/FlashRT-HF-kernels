#!/usr/bin/env python3
"""Correctness tests for speculative-draft-primitives."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "speculative-draft-primitives"
REGISTRATION_INCLUDE = ROOT.parent / "kernels" / "kernel-builder" / "src" / "pyproject" / "templates" / "torch"


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self.ops = getattr(torch.ops, namespace)

    def argmax_bf16(self, logits):
        out = torch.empty((logits.shape[0],), device=logits.device, dtype=torch.int64)
        self.ops.argmax_bf16(logits, out)
        return out

    def accept_greedy_bf16(self, logits, drafts, spec_k):
        argmax = torch.empty((logits.shape[0],), device=logits.device, dtype=torch.int64)
        accept_n = torch.empty((1,), device=logits.device, dtype=torch.int32)
        self.ops.accept_greedy_bf16(logits, drafts, argmax, accept_n, int(spec_k))
        return argmax, accept_n

    def accept_partitioned_bf16(self, logits, drafts, spec_k, parts):
        argmax = torch.empty((logits.shape[0],), device=logits.device, dtype=torch.int64)
        accept_n = torch.empty((1,), device=logits.device, dtype=torch.int32)
        partial_vals = torch.empty((logits.shape[0], parts), device=logits.device, dtype=torch.float32)
        partial_idx = torch.empty((logits.shape[0], parts), device=logits.device, dtype=torch.int32)
        self.ops.accept_partitioned_bf16(
            logits, drafts, argmax, accept_n, partial_vals, partial_idx, int(spec_k), int(parts)
        )
        return argmax, accept_n


def _arch_list() -> str:
    major, minor = torch.cuda.get_device_capability(0)
    return "12.0a" if major >= 12 else f"{major}.{minor}"


def load_source_ops() -> SourceOps:
    from torch.utils.cpp_extension import load

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _arch_list())
    namespace = "speculative_draft_primitives_source_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "speculative_draft_primitives.cu"),
        ],
        extra_include_paths=[str(PACKAGE / "csrc"), str(REGISTRATION_INCLUDE)],
        extra_cflags=["-O3", "-DCUDA_KERNEL"],
        extra_cuda_cflags=[
            "-O3",
            "--expt-relaxed-constexpr",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            "-DCUDA_KERNEL",
        ],
        is_python_module=False,
        verbose=False,
    )
    return SourceOps(namespace)


def load_installed_ops(artifact: str | None):
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return importlib.import_module("speculative_draft_primitives")
    finally:
        if artifact:
            sys.path.remove(artifact)


def _accept_prefix(argmax: torch.Tensor, drafts: torch.Tensor, spec_k: int) -> int:
    a = argmax[:spec_k].cpu()
    d = drafts[:spec_k].cpu()
    n = 0
    for i in range(spec_k):
        if int(a[i]) != int(d[i]):
            break
        n += 1
    return n


def run(ops, mode: str) -> int:
    shapes = [(1, 1024), (16, 32000)] if mode == "smoke" else [
        (1, 1024),
        (4, 4096),
        (16, 32000),
        (16, 248320),
    ]
    count = 0
    for rows, vocab in shapes:
        torch.manual_seed(rows * 1000003 + vocab)
        logits = torch.randn((rows, vocab), device="cuda", dtype=torch.float32).to(torch.bfloat16)
        ref = torch.argmax(logits.float(), dim=1)
        got = ops.argmax_bf16(logits)
        torch.cuda.synchronize()
        if not torch.equal(got.cpu(), ref.cpu()):
            raise AssertionError(f"argmax mismatch rows={rows} vocab={vocab}")
        count += 1

        spec_k = min(rows, 15)
        drafts = ref[:spec_k].clone()
        if spec_k > 2:
            drafts[2:] = (drafts[2:] + 1) % vocab
        got_argmax, accept_n = ops.accept_greedy_bf16(logits, drafts, spec_k)
        torch.cuda.synchronize()
        if not torch.equal(got_argmax.cpu(), ref.cpu()):
            raise AssertionError(f"greedy argmax mismatch rows={rows} vocab={vocab}")
        expected_n = _accept_prefix(ref, drafts, spec_k)
        if int(accept_n.cpu()[0]) != expected_n:
            raise AssertionError(f"greedy accept mismatch got={int(accept_n.cpu()[0])} expected={expected_n}")
        count += 1

        parts = 1 if vocab <= 4096 else 8
        got_argmax, accept_n = ops.accept_partitioned_bf16(logits, drafts, spec_k, parts)
        torch.cuda.synchronize()
        if not torch.equal(got_argmax.cpu(), ref.cpu()):
            raise AssertionError(f"partitioned argmax mismatch rows={rows} vocab={vocab} parts={parts}")
        if int(accept_n.cpu()[0]) != expected_n:
            raise AssertionError("partitioned accept mismatch")
        count += 1

    print(f"speculative-draft-primitives {mode}: passed {count}/{count}")
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    run(ops, args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
