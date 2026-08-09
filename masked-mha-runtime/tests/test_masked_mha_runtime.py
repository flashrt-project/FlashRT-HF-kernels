#!/usr/bin/env python3
"""Strict source and installed-artifact tests for masked-mha-runtime."""

from __future__ import annotations

import argparse
import importlib
import os
import re
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "masked-mha-runtime"
REGISTRATION_INCLUDE = (
    ROOT.parent / "kernels" / "kernel-builder" / "src" / "pyproject"
    / "templates" / "torch"
)


class SourceOps:
    def __init__(self, namespace: str):
        self.ops = getattr(torch.ops, namespace)

    @staticmethod
    def allocate_workspace(q, k):
        stride = (k.shape[0] + 7) // 8 * 8
        return torch.empty(
            (q.shape[1], q.shape[0], stride), device=q.device, dtype=q.dtype
        )

    def forward_static(self, q, k, v, *, logits, out, scale=None):
        scale = q.shape[-1] ** -0.5 if scale is None else scale
        self.ops.forward_static(q, k, v, logits, out, float(scale))
        return out

    def attention_mha_fp16_masked(self, q, k, v, *, logits, out, scale=None):
        scale = q.shape[-1] ** -0.5 if scale is None else scale
        self.ops.attention_mha_fp16_masked(
            q, k, v, logits, out, float(scale)
        )
        return out

    def attention_mha_bf16_masked(
        self, q, k, v, *, logits, out, qkv_token_stride=None, scale=None
    ):
        scale = q.shape[-1] ** -0.5 if scale is None else scale
        if qkv_token_stride is None:
            qkv_token_stride = q.stride(0)
        self.ops.attention_mha_bf16_masked(
            q, k, v, logits, out, float(scale), int(qkv_token_stride)
        )
        return out

    def forward_seqused_static(
        self, q, k, v, valid_k, *, logits, out, scale=None
    ):
        scale = q.shape[-1] ** -0.5 if scale is None else scale
        self.ops.forward_seqused_static(
            q, k, v, valid_k, logits, out, float(scale)
        )
        return out


def load_source_ops():
    from torch.utils.cpp_extension import load

    nvcc = subprocess.check_output(
        ["nvcc", "--version"], text=True
    )
    match = re.search(r"release\s+(\d+)\.", nvcc)
    torch_cuda_major = int(torch.version.cuda.split(".", 1)[0])
    if match and int(match.group(1)) != torch_cuda_major:
        raise RuntimeError(
            "source test requires PyTorch and nvcc from the same CUDA major; "
            f"torch={torch.version.cuda}, nvcc={match.group(1)}.x. Use the "
            "installed artifact or a matching isolated build environment."
        )

    major, minor = torch.cuda.get_device_capability(0)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    namespace = "masked_mha_runtime_source_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "attention_mha_masked.cu"),
            str(PACKAGE / "csrc" / "attention_seqused_fused.cu"),
        ],
        extra_include_paths=[str(PACKAGE / "csrc"), str(REGISTRATION_INCLUDE)],
        extra_cflags=["-O3", "-DCUDA_KERNEL"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-DCUDA_KERNEL"],
        extra_ldflags=["-lcublas"],
        is_python_module=False,
        verbose=False,
    )
    return SourceOps(namespace)


def load_installed_ops(artifact):
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return importlib.import_module("masked_mha_runtime")
    finally:
        if artifact:
            sys.path.remove(artifact)


def metrics(got, ref):
    diff = (got.float() - ref.float()).abs()
    cosine = torch.nn.functional.cosine_similarity(
        got.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    return float(diff.max()), float(torch.quantile(diff.flatten(), 0.99)), float(cosine)


def run_case(ops, dtype, sq, sk, heads, dim, fused_stride=False):
    torch.manual_seed(1000 + sq + sk + dim)
    if fused_stride:
        packed = torch.randn((sk, 3, heads, dim), device="cuda", dtype=dtype)
        q = packed[:sq, 0]
        k = packed[:, 1]
        v = packed[:, 2]
    else:
        q = torch.randn((sq, heads, dim), device="cuda", dtype=dtype)
        k = torch.randn((sk, heads, dim), device="cuda", dtype=dtype)
        v = torch.randn_like(k)
    logits = ops.allocate_workspace(q, k)
    logits.fill_(float("nan"))
    out = torch.empty_like(q, memory_format=torch.contiguous_format)
    got = ops.forward_static(q, k, v, logits=logits, out=out)
    torch.cuda.synchronize()
    alias_logits = ops.allocate_workspace(q, k)
    alias_logits.fill_(float("nan"))
    alias_out = torch.empty_like(q, memory_format=torch.contiguous_format)
    if dtype is torch.float16:
        alias_got = ops.attention_mha_fp16_masked(
            q, k, v, logits=alias_logits, out=alias_out
        )
    else:
        alias_got = ops.attention_mha_bf16_masked(
            q, k, v, logits=alias_logits, out=alias_out,
            qkv_token_stride=q.stride(0),
        )
    torch.cuda.synchronize()
    if not torch.equal(alias_got, got):
        raise AssertionError("explicit masked MHA entry differs from forward_static")
    if dtype is torch.bfloat16:
        try:
            ops.attention_mha_bf16_masked(
                q, k, v, logits=alias_logits, out=alias_out,
                qkv_token_stride=q.stride(0) + 1,
            )
        except RuntimeError as exc:
            if "qkv_token_stride" not in str(exc):
                raise
        else:
            raise AssertionError("invalid qkv_token_stride was not rejected")
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.permute(1, 0, 2).unsqueeze(0).float(),
        k.permute(1, 0, 2).unsqueeze(0).float(),
        v.permute(1, 0, 2).unsqueeze(0).float(),
    ).squeeze(0).permute(1, 0, 2).to(dtype)
    max_abs, p99_abs, cosine = metrics(got, ref)
    atol = 0.00390625 if dtype is torch.float16 else 0.015625
    if not torch.isfinite(got.float()).all() or cosine < 0.999 or p99_abs > atol:
        raise AssertionError(
            f"dtype={dtype} sq={sq} sk={sk} h={heads} d={dim}: "
            f"max={max_abs} p99={p99_abs} cos={cosine}"
        )

    # Preserve fused-QKV token strides. Cloning each view separately can
    # normalize a size-one query dimension and destroy the shared stride.
    static_q = q
    static_k = k
    static_v = v
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        if dtype is torch.float16:
            ops.attention_mha_fp16_masked(
                static_q, static_k, static_v, logits=logits, out=out
            )
        else:
            ops.attention_mha_bf16_masked(
                static_q, static_k, static_v, logits=logits, out=out,
                qkv_token_stride=static_q.stride(0),
            )
    graph.replay()
    first = out.clone()
    graph.replay()
    torch.cuda.synchronize()
    if not torch.equal(out, first):
        raise AssertionError("CUDA Graph replay is not bitwise deterministic")
    print(
        f"PASS {dtype} sq={sq} sk={sk} h={heads} d={dim} "
        f"fused_stride={fused_stride} max={max_abs:.6f} "
        f"p99={p99_abs:.6f} cos={cosine:.8f}"
    )


def run_seqused_case(ops, sq, sk, valid, heads=8, dim=256):
    torch.manual_seed(164000 + sq + sk + valid)
    q = torch.randn((sq, heads, dim), device="cuda", dtype=torch.float16)
    k = torch.randn((sk, dim), device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    valid_k = torch.tensor([valid], device="cuda", dtype=torch.int32)
    logits = torch.full(
        (sq * heads, sk), float("nan"), device="cuda", dtype=torch.float16
    )
    out = torch.empty_like(q)
    got = ops.forward_seqused_static(
        q, k, v, valid_k, logits=logits, out=out
    )
    torch.cuda.synchronize()
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.permute(1, 0, 2).unsqueeze(0).float(),
        k[:valid].unsqueeze(0).unsqueeze(0).float(),
        v[:valid].unsqueeze(0).unsqueeze(0).float(),
        enable_gqa=True,
    ).squeeze(0).permute(1, 0, 2).half()
    max_abs, p99_abs, cosine = metrics(got, ref)
    if not torch.isfinite(got.float()).all() or cosine < 0.999 or p99_abs > 0.00390625:
        raise AssertionError(
            f"seqused sq={sq} sk={sk} valid={valid} h={heads} d={dim}: "
            f"max={max_abs} p99={p99_abs} cos={cosine}"
        )
    if not torch.equal(logits[:, valid:], torch.zeros_like(logits[:, valid:])):
        raise AssertionError("seqused probabilities beyond valid_k are not zero")

    residual = torch.randn_like(out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.forward_seqused_static(
            q, k, v, valid_k, logits=logits, out=out
        )
    graph.replay()
    first = out.clone()
    graph.replay()
    torch.cuda.synchronize()
    if not torch.equal(first, out):
        raise AssertionError("seqused CUDA Graph replay is not bitwise deterministic")
    del residual
    print(
        f"PASS seqused sq={sq} sk={sk} valid={valid} h={heads} d={dim} "
        f"max={max_abs:.6f} p99={p99_abs:.6f} cos={cosine:.8f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    cases = [
        (torch.float16, 41, 41, 32, 48, False),
        (torch.bfloat16, 41, 41, 32, 48, True),
    ]
    if args.mode == "full":
        cases.extend([
            (torch.float16, 1, 277, 16, 128, False),
            (torch.bfloat16, 1, 1024, 1, 16, True),
            (torch.bfloat16, 1, 1025, 1, 16, True),
            (torch.bfloat16, 1, 2048, 1, 16, True),
        ])
    for case in cases:
        run_case(ops, *case)
    seqused_cases = [(10, 456, 456)]
    if args.mode == "full":
        seqused_cases.extend([(10, 968, 456), (10, 968, 712), (10, 968, 968)])
    for case in seqused_cases:
        run_seqused_case(ops, *case)
    total = len(cases) + len(seqused_cases)
    print(f"masked-mha-runtime {args.backend} {args.mode}: passed {total}/{total}")


if __name__ == "__main__":
    main()
