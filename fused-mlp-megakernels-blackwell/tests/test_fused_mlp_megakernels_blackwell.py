#!/usr/bin/env python3
"""Strict correctness and graph tests for Blackwell FP16 MLP megakernels."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "fused-mlp-megakernels-blackwell"
REGISTRATION_INCLUDE = (
    ROOT.parent / "kernels" / "kernel-builder" / "src" / "pyproject"
    / "templates" / "torch"
)
CUTLASS = ROOT / ".." / "official" / "FlashRT" / "third_party" / "cutlass"


class SourceOps:
    def __init__(self, namespace: str):
        self.ops = getattr(torch.ops, namespace)

    def fp16_geglu_fused(
        self, input, gate_weight, up_weight, *, gate_scratch=None, output=None
    ):
        shape = (input.shape[0], gate_weight.shape[0])
        if gate_scratch is None:
            gate_scratch = torch.empty(shape, device=input.device, dtype=torch.float16)
        if output is None:
            output = torch.empty(shape, device=input.device, dtype=torch.float16)
        self.ops.fp16_geglu_fused_out(
            input, gate_weight, up_weight, gate_scratch, output
        )
        return output


def load_source_ops():
    from torch.utils.cpp_extension import load

    major, minor = torch.cuda.get_device_capability(0)
    if major * 10 + minor not in (100, 103, 110):
        raise RuntimeError("source test requires SM100, SM103, or SM110")
    arch = f"{major}.{minor}a" if major in (10, 12) else f"{major}.{minor}a"
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", arch)
    namespace = "fused_mlp_megakernels_blackwell_source_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "mega" / "flashrt_megakernel_geglu.cu"),
        ],
        extra_include_paths=[
            str(PACKAGE / "csrc" / "mega"),
            str(CUTLASS / "include"),
            str(CUTLASS / "tools" / "util" / "include"),
            str(REGISTRATION_INCLUDE),
        ],
        extra_cflags=["-O3", "-DCUDA_KERNEL"],
        extra_cuda_cflags=[
            "-O3", "--use_fast_math", "--expt-relaxed-constexpr", "-DCUDA_KERNEL"
        ],
        verbose=False,
    )
    return SourceOps(namespace)


def load_installed(artifact):
    if artifact:
        sys.path.insert(0, artifact)
    return importlib.import_module("fused_mlp_megakernels_blackwell")


def reference(input, gate_weight, up_weight):
    gate = F.gelu(input @ gate_weight.t(), approximate="tanh")
    return (gate * (input @ up_weight.t())).to(torch.float16)


def metrics(actual, expected):
    diff = (actual.float() - expected.float()).abs().flatten()
    cosine = F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0)
    return float(diff.max()), float(torch.quantile(diff, 0.99)), float(diff.mean()), float(cosine)


def run_row(ops, m, n, k, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    input = (torch.randn((m, k), device="cuda", generator=gen) * 0.15).half()
    gate = (torch.randn((n, k), device="cuda", generator=gen) * 0.05).half()
    up = (torch.randn((n, k), device="cuda", generator=gen) * 0.05).half()
    expected = reference(input, gate, up)
    actual = ops.fp16_geglu_fused(input, gate, up)
    torch.cuda.synchronize()
    maximum, p99, mean, cosine = metrics(actual, expected)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    assert cosine >= 0.9995, (m, n, k, maximum, p99, mean, cosine)
    return maximum, p99, mean, cosine


def run_compile_and_graph(ops):
    m = n = k = 128
    input = torch.randn((m, k), device="cuda", dtype=torch.float16)
    gate = torch.randn((n, k), device="cuda", dtype=torch.float16) * 0.02
    up = torch.randn((n, k), device="cuda", dtype=torch.float16) * 0.02
    eager = ops.fp16_geglu_fused(input, gate, up)
    compiled = torch.compile(ops.fp16_geglu_fused, fullgraph=True)(input, gate, up)
    torch.testing.assert_close(compiled, eager, rtol=0, atol=0)

    scratch = torch.empty((m, n), device="cuda", dtype=torch.float16)
    output = torch.empty_like(scratch)
    # Warm once so the upstream CUTLASS workspace is initialized before capture.
    ops.fp16_geglu_fused(input, gate, up, gate_scratch=scratch, output=output)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.fp16_geglu_fused(
            input, gate, up, gate_scratch=scratch, output=output
        )
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, eager, rtol=0, atol=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("source", "installed"), default="source")
    parser.add_argument("--artifact")
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    ops = load_source_ops() if args.backend == "source" else load_installed(args.artifact)
    shapes = [(128, 128, 128)]
    if args.mode == "full":
        shapes += [
            (64, 256, 256), (127, 256, 256), (129, 256, 256),
            (256, 1024, 1024), (768, 2048, 2048),
            (768, 16384, 2048),
        ]
    for index, shape in enumerate(shapes):
        values = run_row(ops, *shape, seed=2026 + index)
        print(f"M={shape[0]} N={shape[1]} K={shape[2]} max={values[0]:.8f} "
              f"p99={values[1]:.8f} mean={values[2]:.8f} cos={values[3]:.8f}")
    run_compile_and_graph(ops)
    print(f"PASS fused-mlp-megakernels-blackwell {args.backend}: "
          f"{len(shapes)} numeric rows + compile/graph")


if __name__ == "__main__":
    main()
