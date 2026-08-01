#!/usr/bin/env python3
"""Strict correctness tests for audio-codebook-primitives."""

from __future__ import annotations

import argparse
import importlib
import math
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "audio-codebook-primitives"
REGISTRATION_INCLUDE = (
    ROOT.parent / "kernels" / "kernel-builder" / "src" / "pyproject"
    / "templates" / "torch"
)
MASK64 = (1 << 64) - 1


class SourceOps:
    def __init__(self, namespace: str):
        self.ops = getattr(torch.ops, namespace)

    def delayed_codebook_argmax_embed_bf16(
        self, logits, codebook, *, delay, boc, codes=None, embedding=None
    ):
        codes, embedding = make_outputs(logits, codebook, codes, embedding)
        self.ops.delayed_codebook_argmax_embed_bf16(
            logits, codebook, delay, boc, codes, embedding
        )
        return codes, embedding

    def delayed_codebook_sample_embed_bf16(
        self, logits, codebook, *, delay, boc, temperature, seed, step,
        codes=None, embedding=None
    ):
        codes, embedding = make_outputs(logits, codebook, codes, embedding)
        self.ops.delayed_codebook_sample_embed_bf16(
            logits, codebook, delay, boc, temperature, seed, step,
            codes, embedding
        )
        return codes, embedding


def make_outputs(logits, codebook, codes=None, embedding=None):
    if codes is None:
        codes = torch.empty(logits.shape[0], device="cuda", dtype=torch.int64)
    if embedding is None:
        embedding = torch.empty(codebook.shape[2], device="cuda", dtype=torch.bfloat16)
    return codes, embedding


def load_source_ops():
    from torch.utils.cpp_extension import load

    major, minor = torch.cuda.get_device_capability(0)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a" if major >= 12 else f"{major}.{minor}")
    namespace = "audio_codebook_primitives_source_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "delayed_codebook_kernels.cu"),
        ],
        extra_include_paths=[str(PACKAGE / "csrc"), str(REGISTRATION_INCLUDE)],
        extra_cflags=["-O3", "-DCUDA_KERNEL"],
        extra_cuda_cflags=["-O3", "-DCUDA_KERNEL"],
        verbose=False,
    )
    return SourceOps(namespace)


def load_installed(artifact):
    if artifact:
        sys.path.insert(0, artifact)
    return importlib.import_module("audio_codebook_primitives")


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return (x ^ (x >> 31)) & MASK64


def uniform_open01(seed: int, step: int, codebook: int) -> float:
    key = (
        seed
        ^ ((step * 0xD2B74407B1CE6E93) & MASK64)
        ^ ((codebook * 0xCA5A826395121157) & MASK64)
    ) & MASK64
    bits = splitmix64(key) >> 40
    return (float(bits) + 0.5) * (2.0 ** -24)


def embedding_reference(codebook, codes):
    acc = torch.zeros(codebook.shape[2], device="cuda", dtype=torch.float32)
    for cb in range(codebook.shape[0]):
        acc = acc + codebook[cb, int(codes[cb])].float()
    return acc.to(torch.bfloat16)


def stats(actual, expected):
    diff = (actual.float() - expected.float()).abs().flatten()
    return (
        float(diff.max()),
        float(torch.quantile(diff, 0.99)),
        float(diff.mean()),
        float(torch.nn.functional.cosine_similarity(
            actual.float().flatten(), expected.float().flatten(), dim=0
        )),
    )


def run_argmax(ops, c, v, h, delay, boc, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn((c, v), device="cuda", generator=gen, dtype=torch.bfloat16)
    codebook = (torch.randn((c, v, h), device="cuda", generator=gen) * 0.1).bfloat16()
    # Exercise the documented lowest-index tie break.
    logits[0, 3] = logits[0, 7] = torch.tensor(20.0, device="cuda", dtype=torch.bfloat16)
    expected_codes = logits.float().argmax(dim=1).to(torch.int64)
    if delay < c:
        expected_codes[delay + 1 :] = boc
    codes, embedding = ops.delayed_codebook_argmax_embed_bf16(
        logits, codebook, delay=delay, boc=boc
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(codes, expected_codes, rtol=0, atol=0)
    expected_embedding = embedding_reference(codebook, expected_codes)
    torch.testing.assert_close(embedding, expected_embedding, rtol=0, atol=0)
    return stats(embedding, expected_embedding)


def run_sample_equal_logits(ops, c, v, h, delay, boc, seed, step):
    logits = torch.zeros((c, v), device="cuda", dtype=torch.bfloat16)
    codebook = torch.arange(c * v * h, device="cuda", dtype=torch.float32).reshape(c, v, h)
    codebook = ((codebook.remainder(31) - 15) / 64).to(torch.bfloat16)
    expected = torch.tensor(
        [min(v - 1, math.floor(uniform_open01(seed, step, cb) * v)) for cb in range(c)],
        device="cuda", dtype=torch.int64,
    )
    if delay < c:
        expected[delay + 1 :] = boc
    codes, embedding = ops.delayed_codebook_sample_embed_bf16(
        logits, codebook, delay=delay, boc=boc, temperature=0.73,
        seed=seed, step=step
    )
    codes_2, embedding_2 = ops.delayed_codebook_sample_embed_bf16(
        logits, codebook, delay=delay, boc=boc, temperature=0.73,
        seed=seed, step=step
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(codes, expected, rtol=0, atol=0)
    torch.testing.assert_close(codes_2, codes, rtol=0, atol=0)
    torch.testing.assert_close(embedding_2, embedding, rtol=0, atol=0)
    expected_embedding = embedding_reference(codebook, expected)
    torch.testing.assert_close(embedding, expected_embedding, rtol=0, atol=0)
    return stats(embedding, expected_embedding)


def run_compile_and_graph(ops):
    c, v, h = 8, 1026, 1024
    logits = torch.randn((c, v), device="cuda", dtype=torch.bfloat16)
    codebook = torch.randn((c, v, h), device="cuda", dtype=torch.bfloat16)

    def invoke(a, b):
        return ops.delayed_codebook_argmax_embed_bf16(a, b, delay=7, boc=1024)

    eager = invoke(logits, codebook)
    compiled = torch.compile(invoke, fullgraph=True)(logits, codebook)
    torch.testing.assert_close(compiled[0], eager[0], rtol=0, atol=0)
    torch.testing.assert_close(compiled[1], eager[1], rtol=0, atol=0)

    codes = torch.empty(c, device="cuda", dtype=torch.int64)
    embedding = torch.empty(h, device="cuda", dtype=torch.bfloat16)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.delayed_codebook_argmax_embed_bf16(
            logits, codebook, delay=7, boc=1024,
            codes=codes, embedding=embedding
        )
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(codes, eager[0], rtol=0, atol=0)
    torch.testing.assert_close(embedding, eager[1], rtol=0, atol=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("source", "installed"), default="source")
    parser.add_argument("--artifact")
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    ops = load_source_ops() if args.backend == "source" else load_installed(args.artifact)
    rows = [
        ("argmax_small", run_argmax(ops, 1, 16, 32, 1, 3, 101)),
        ("argmax_higgs_delay0", run_argmax(ops, 8, 1026, 1024, 0, 1024, 102)),
        ("sample_higgs", run_sample_equal_logits(ops, 8, 1026, 1024, 7, 1024, 2026, 17)),
    ]
    if args.mode == "full":
        rows += [
            ("argmax_higgs_full", run_argmax(ops, 8, 1026, 1024, 8, 1024, 103)),
            ("argmax_nonpower_hidden", run_argmax(ops, 5, 1025, 1023, 3, 1024, 104)),
            ("sample_delay0", run_sample_equal_logits(ops, 8, 1026, 257, 0, 1024, 77, 0)),
            ("sample_step1", run_sample_equal_logits(ops, 8, 1026, 257, 8, 1024, 77, 1)),
        ]
    run_compile_and_graph(ops)
    for name, (maximum, p99, mean, cosine) in rows:
        print(f"{name}: max={maximum:.8f} p99={p99:.8f} mean={mean:.8f} cos={cosine:.8f}")
    print(f"PASS audio-codebook-primitives {args.backend}: {len(rows)} numeric rows + compile/graph")


if __name__ == "__main__":
    main()
