#!/usr/bin/env python3
"""Correctness, compile, and graph gates for FP8 GQA cross-attention."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "fp8-cross-attention-blackwell"
REGISTRATION_INCLUDE = (
    ROOT.parent / "kernels" / "kernel-builder" / "src" / "pyproject"
    / "templates" / "torch"
)
CUTLASS = ROOT.parent / "official" / "FlashRT" / "third_party" / "cutlass"


class SourceOps:
    def __init__(self, namespace):
        self.ops = getattr(torch.ops, namespace)

    def fp8_gqa_cross_attention_bf16(
        self, q, k, v, *, query_scale, key_scale, value_scale,
        output=None, lse=None, workspace=None
    ):
        output = output if output is not None else torch.empty_like(q, dtype=torch.bfloat16)
        sqr = (q.shape[1] + 127) // 128 * 128
        lse = lse if lse is not None else torch.empty(
            (q.shape[0], q.shape[2], sqr), device=q.device, dtype=torch.float32
        )
        workspace = workspace if workspace is not None else torch.empty(
            4 * 1024 * 1024, device=q.device, dtype=torch.uint8
        )
        self.ops.fp8_gqa_cross_attention_bf16_out(
            q, k, v, query_scale, key_scale, value_scale,
            output, lse, workspace
        )
        return output


def load_source_ops():
    from torch.utils.cpp_extension import load

    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor not in (100, 103, 110):
        raise RuntimeError("source tests require SM100, SM103, or SM110")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}a")
    namespace = "fp8_cross_attention_blackwell_source_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "cutlass_fp8_gqa_cross_attention_sm100.cu"),
        ],
        extra_include_paths=[
            str(PACKAGE / "csrc" / "fmha77"),
            str(CUTLASS / "include"),
            str(CUTLASS / "tools" / "util" / "include"),
            str(REGISTRATION_INCLUDE),
        ],
        extra_cflags=["-O3", "-DCUDA_KERNEL"],
        extra_cuda_cflags=[
            "-O3", "--expt-relaxed-constexpr", "-DCUDA_KERNEL",
            "-DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1",
        ],
        verbose=False,
    )
    return SourceOps(namespace)


def load_installed(artifact):
    if artifact:
        sys.path.insert(0, artifact)
    return importlib.import_module("fp8_cross_attention_blackwell")


def quantize(x):
    scale = max(float(x.abs().max()) / 448.0, 1e-6)
    return (x / scale).clamp(-448, 448).to(torch.float8_e4m3fn), scale


def reference(q, k, v, qs, ks, vs):
    qf = (q.float() * qs).transpose(1, 2)
    kf = (k.float() * ks).transpose(1, 2)
    vf = (v.float() * vs).transpose(1, 2)
    groups = qf.shape[1] // kf.shape[1]
    kf = kf.repeat_interleave(groups, dim=1)
    vf = vf.repeat_interleave(groups, dim=1)
    return F.scaled_dot_product_attention(qf, kf, vf).transpose(1, 2).bfloat16()


def run_row(ops, b, sq, sk, hq, hkv, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q, qs = quantize(torch.randn((b, sq, hq, 128), device="cuda", generator=gen) * 0.2)
    k, ks = quantize(torch.randn((b, sk, hkv, 128), device="cuda", generator=gen) * 0.2)
    v, vs = quantize(torch.randn((b, sk, hkv, 128), device="cuda", generator=gen) * 0.2)
    expected = reference(q, k, v, qs, ks, vs)
    actual = ops.fp8_gqa_cross_attention_bf16(
        q, k, v, query_scale=qs, key_scale=ks, value_scale=vs
    )
    torch.cuda.synchronize()
    diff = (actual.float() - expected.float()).abs().flatten()
    maximum = float(diff.max())
    p99 = float(torch.quantile(diff, 0.99))
    mean = float(diff.mean())
    cosine = float(F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0))
    assert actual.dtype == torch.bfloat16
    assert maximum <= 0.004 and cosine >= 0.9995 and p99 <= 0.002 and mean <= 0.0005, (
        b, sq, sk, hq, hkv, maximum, p99, mean, cosine
    )
    return maximum, p99, mean, cosine


def run_graph(ops):
    b, sq, sk, hq, hkv = 1, 51, 257, 16, 4
    q, qs = quantize(torch.randn((b, sq, hq, 128), device="cuda"))
    k, ks = quantize(torch.randn((b, sk, hkv, 128), device="cuda"))
    v, vs = quantize(torch.randn((b, sk, hkv, 128), device="cuda"))
    output = torch.empty_like(q, dtype=torch.bfloat16)
    lse = torch.empty((b, hq, 128), device="cuda", dtype=torch.float32)
    workspace = torch.empty(4 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    eager = ops.fp8_gqa_cross_attention_bf16(
        q, k, v, query_scale=qs, key_scale=ks, value_scale=vs,
        output=output, lse=lse, workspace=workspace
    ).clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.fp8_gqa_cross_attention_bf16(
            q, k, v, query_scale=qs, key_scale=ks, value_scale=vs,
            output=output, lse=lse, workspace=workspace
        )
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, eager, rtol=0, atol=0)


def run_compile(ops):
    b, sq, sk, hq, hkv = 1, 129, 257, 16, 4
    q, qs = quantize(torch.randn((b, sq, hq, 128), device="cuda") * 0.2)
    k, ks = quantize(torch.randn((b, sk, hkv, 128), device="cuda") * 0.2)
    v, vs = quantize(torch.randn((b, sk, hkv, 128), device="cuda") * 0.2)

    def call(q_, k_, v_):
        return ops.fp8_gqa_cross_attention_bf16(
            q_, k_, v_, query_scale=qs, key_scale=ks, value_scale=vs
        )

    eager = call(q, k, v)
    compiled = torch.compile(call, fullgraph=True)(q, k, v)
    torch.testing.assert_close(compiled, eager, rtol=0, atol=0)


def run_rejections(ops):
    q = torch.zeros((1, 8, 3, 128), device="cuda", dtype=torch.float8_e4m3fn)
    k = torch.zeros((1, 8, 2, 128), device="cuda", dtype=torch.float8_e4m3fn)
    try:
        ops.fp8_gqa_cross_attention_bf16(
            q, k, k, query_scale=1.0, key_scale=1.0, value_scale=1.0
        )
    except RuntimeError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("non-divisible GQA heads must be rejected")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("source", "installed"), default="source")
    parser.add_argument("--artifact")
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    args = parser.parse_args()
    ops = load_source_ops() if args.backend == "source" else load_installed(args.artifact)
    shapes = [(1, 51, 257, 16, 4), (1, 128, 128, 28, 4)]
    if args.mode == "full":
        shapes += [
            (1, 1, 1, 8, 1), (1, 1, 127, 8, 1),
            (1, 127, 129, 32, 8), (1, 128, 255, 32, 8),
            (1, 129, 256, 32, 8), (2, 129, 513, 16, 4),
            (1, 786, 7984, 28, 4),
        ]
    for index, shape in enumerate(shapes):
        metrics = run_row(ops, *shape, seed=2026 + index)
        print(f"shape={shape} max={metrics[0]:.8f} p99={metrics[1]:.8f} "
              f"mean={metrics[2]:.8f} cos={metrics[3]:.8f}")
    run_graph(ops)
    run_rejections(ops)
    if args.backend == "installed":
        run_compile(ops)
    print(f"PASS fp8-cross-attention-blackwell {args.backend}: "
          f"{len(shapes)} rows + rejection/graph"
          f"{'/compile' if args.backend == 'installed' else ''}")


if __name__ == "__main__":
    main()
