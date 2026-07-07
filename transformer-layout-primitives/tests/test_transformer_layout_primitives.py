#!/usr/bin/env python3
"""Correctness tests for transformer-layout-primitives."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "transformer-layout-primitives"
REGISTRATION_INCLUDE = ROOT.parent / "kernels" / "kernel-builder" / "src" / "pyproject" / "templates" / "torch"


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self.ops = getattr(torch.ops, namespace)

    def fill_neginf_bf16(self, dst):
        self.ops.fill_neginf_bf16(dst)
        return dst

    def add_bias_bf16_(self, data, bias):
        self.ops.add_bias_bf16_(data, bias)
        return data

    def repeat_interleave_heads_bf16(self, src, repeat, out=None):
        if out is None:
            out = torch.empty((src.shape[0], src.shape[1] * repeat, src.shape[2]), device=src.device, dtype=src.dtype)
        self.ops.repeat_interleave_heads_bf16(src, int(repeat), out)
        return out

    def text_gather_bf16(self, src, batch, seq, out=None):
        if out is None:
            out = torch.empty((2 * batch, src.shape[1]), device=src.device, dtype=src.dtype)
        self.ops.text_gather_bf16(src, int(batch), int(seq), out)
        return out

    def text_scatter_bf16(self, dst, src, batch, seq):
        self.ops.text_scatter_bf16(dst, src, int(batch), int(seq))
        return dst

    def rope_rotate_half_bf16_(self, x, cos, sin):
        self.ops.rope_rotate_half_bf16_(x, cos, sin)
        return x

    def qk_rmsnorm_rope_bf16_(self, qk, weight, cos, sin, eps=1e-6):
        self.ops.qk_rmsnorm_rope_bf16_(qk, weight, cos, sin, float(eps))
        return qk


def _arch_list() -> str:
    major, minor = torch.cuda.get_device_capability(0)
    return "12.0a" if major >= 12 else f"{major}.{minor}"


def load_source_ops() -> SourceOps:
    from torch.utils.cpp_extension import load

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _arch_list())
    namespace = "transformer_layout_primitives_source_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "transformer_layout_primitives.cu"),
        ],
        extra_include_paths=[str(PACKAGE / "csrc"), str(REGISTRATION_INCLUDE)],
        extra_cflags=["-O3", "-DCUDA_KERNEL"],
        extra_cuda_cflags=[
            "-O3",
            "--expt-relaxed-constexpr",
            "-DCUDA_KERNEL",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        ],
        is_python_module=False,
        verbose=False,
    )
    return SourceOps(namespace)


def load_installed_ops(artifact: str | None):
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return importlib.import_module("transformer_layout_primitives")
    finally:
        if artifact:
            sys.path.remove(artifact)


def rotate_half_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    lo = x[..., :half].float()
    hi = x[..., half:].float()
    c = cos[:, None, :half].float()
    s = sin[:, None, :half].float()
    return torch.cat([lo * c - hi * s, hi * c + lo * s], dim=-1).to(torch.bfloat16)


def qk_rmsnorm_rope_ref(
    qk: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    rms = torch.rsqrt((qk.float() * qk.float()).mean(dim=-1, keepdim=True) + eps)
    normed = (qk.float() * rms * weight.float()).to(torch.bfloat16)
    return rotate_half_ref(normed, cos, sin)


def metrics(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float]:
    diff = (got.float() - ref.float()).abs()
    cos = torch.nn.functional.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0).item()
    return float(diff.max().item()), float(diff.mean().item()), float(cos)


def assert_close(name: str, got: torch.Tensor, ref: torch.Tensor, atol: float, cos_min: float) -> None:
    max_abs, mean_abs, cos = metrics(got, ref)
    print(f"{name}: max_abs={max_abs:.6f} mean_abs={mean_abs:.6f} cosine={cos:.8f}")
    if max_abs > atol or cos < cos_min:
        raise AssertionError(f"{name} failed: max_abs={max_abs} cosine={cos}")


def run(ops, mode: str) -> int:
    torch.manual_seed(31)
    count = 0
    layout_shapes = [(2, 5, 128), (3, 49, 256)] if mode == "smoke" else [
        (1, 1, 128),
        (2, 5, 128),
        (3, 49, 256),
        (4, 256, 1024),
        (2, 2520, 2048),
    ]
    for batch, seq, dim in layout_shapes:
        x = torch.randn((batch * seq, dim), device="cuda", dtype=torch.bfloat16)

        dst = torch.empty_like(x)
        ops.fill_neginf_bf16(dst)
        ref_neginf = torch.full_like(x, -1e30)
        torch.testing.assert_close(dst.float().cpu(), ref_neginf.float().cpu(), rtol=0, atol=0)
        count += 1

        bias = torch.randn((dim,), device="cuda", dtype=torch.bfloat16)
        got = x.clone()
        ops.add_bias_bf16_(got, bias)
        ref = (x.float() + bias.float()).to(torch.bfloat16)
        torch.testing.assert_close(got.cpu(), ref.cpu(), rtol=0, atol=0)
        count += 1

        gathered = ops.text_gather_bf16(x, batch, seq)
        ref_gather = torch.stack([x[b * seq + offset] for b in range(batch) for offset in (0, seq - 1)], dim=0)
        torch.testing.assert_close(gathered.cpu(), ref_gather.cpu(), rtol=0, atol=0)
        count += 1

        scattered = torch.zeros_like(x)
        ops.text_scatter_bf16(scattered, gathered, batch, seq)
        ref_scatter = torch.zeros_like(x)
        for b in range(batch):
            ref_scatter[b * seq] = gathered[2 * b]
            ref_scatter[b * seq + seq - 1] = gathered[2 * b + 1]
        torch.testing.assert_close(scattered.cpu(), ref_scatter.cpu(), rtol=0, atol=0)
        count += 1

    repeat_shapes = [(17, 4, 64, 2), (128, 8, 128, 4)] if mode == "smoke" else [
        (1, 1, 64, 8),
        (17, 4, 64, 2),
        (128, 8, 128, 4),
        (2520, 8, 128, 4),
    ]
    for seq, heads, dim, repeat in repeat_shapes:
        src = torch.randn((seq, heads, dim), device="cuda", dtype=torch.bfloat16)
        got = ops.repeat_interleave_heads_bf16(src, repeat)
        ref = src.repeat_interleave(repeat, dim=1)
        torch.testing.assert_close(got.cpu(), ref.cpu(), rtol=0, atol=0)
        count += 1

    rope_shapes = [(17, 4, 64), (128, 8, 128)] if mode == "smoke" else [
        (1, 1, 64),
        (17, 4, 64),
        (128, 8, 128),
        (2520, 32, 128),
    ]
    for seq, heads, dim in rope_shapes:
        x = torch.randn((seq, heads, dim), device="cuda", dtype=torch.bfloat16)
        cos = torch.randn((seq, dim), device="cuda", dtype=torch.bfloat16)
        sin = torch.randn((seq, dim), device="cuda", dtype=torch.bfloat16)
        got = x.clone()
        ops.rope_rotate_half_bf16_(got, cos, sin)
        ref = rotate_half_ref(x, cos, sin)
        assert_close(f"rope seq={seq} heads={heads} dim={dim}", got, ref, atol=0, cos_min=0.999999)
        count += 1

        weight = torch.randn((dim,), device="cuda", dtype=torch.bfloat16)
        got = x.clone()
        ops.qk_rmsnorm_rope_bf16_(got, weight, cos, sin)
        ref = qk_rmsnorm_rope_ref(x, weight, cos, sin)
        assert_close(f"qk_rmsnorm_rope seq={seq} heads={heads} dim={dim}", got, ref, atol=0.015625, cos_min=0.999999)
        count += 1

    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    count = run(ops, args.mode)
    print(f"transformer-layout-primitives {args.backend} {args.mode}: passed {count}/{count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
