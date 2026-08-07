#!/usr/bin/env python3
"""Correctness tests for flashrt-vla-video (Q/K RMSNorm + RoPE, packed split)."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "flashrt-vla-video"
REGISTRATION_INCLUDE = (
    ROOT.parent
    / "kernels"
    / "kernel-builder"
    / "src"
    / "pyproject"
    / "templates"
    / "torch"
)


def _arch_list() -> str:
    major, minor = torch.cuda.get_device_capability(0)
    if major >= 12:
        return "12.0a"
    if (major, minor) == (11, 0):
        return "11.0a"
    return f"{major}.{minor}"


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self._ops = getattr(torch.ops, namespace)

    def q_norm_rope(self, q, weight, cos, sin, *, eps=1e-6, out=None):
        if out is None:
            out = torch.empty_like(q)
        self._ops.q_norm_rope_bf16(q, weight, cos, sin, out, float(eps))
        return out

    def k_norm_rope_v_cache(self, k, v, weight, cos, sin, *, eps=1e-6, k_out=None, v_out=None):
        if k_out is None:
            k_out = torch.empty_like(k)
        if v_out is None:
            v_out = torch.empty_like(v)
        self._ops.k_norm_rope_v_cache_bf16(k, v, weight, cos, sin, k_out, v_out, float(eps))
        return k_out, v_out

    def qkv_split_norm_rope(
        self, packed, qw, kw, freqs_re, freqs_im, *, heads, head_dim, seq_len=None,
        q_out=None, k_out=None, eps=1e-6,
    ):
        if seq_len is None:
            seq_len = packed.shape[1]
        if q_out is None:
            q_out = torch.empty((packed.shape[0], packed.shape[1], heads, head_dim),
                                device=packed.device, dtype=packed.dtype)
        if k_out is None:
            k_out = torch.empty_like(q_out)
        self._ops.qkv_split_norm_rope_bf16(
            packed, qw, kw, freqs_re, freqs_im, q_out, k_out,
            int(heads), int(head_dim), int(seq_len), float(eps),
        )
        return q_out, k_out


class InstalledOps:
    def __init__(self, module) -> None:
        self._module = module

    def q_norm_rope(self, q, weight, cos, sin, *, eps=1e-6, out=None):
        return self._module.q_norm_rope_bf16(q, weight, cos, sin, out=out, eps=eps)

    def k_norm_rope_v_cache(self, k, v, weight, cos, sin, *, eps=1e-6, k_out=None, v_out=None):
        return self._module.k_norm_rope_v_cache_bf16(
            k, v, weight, cos, sin, k_out=k_out, v_out=v_out, eps=eps
        )

    def qkv_split_norm_rope(
        self, packed, qw, kw, freqs_re, freqs_im, *, heads, head_dim, seq_len=None,
        q_out=None, k_out=None, eps=1e-6,
    ):
        return self._module.qkv_split_norm_rope_bf16(
            packed, qw, kw, freqs_re, freqs_im, heads=heads, head_dim=head_dim,
            seq_len=seq_len, q_out=q_out, k_out=k_out, eps=eps,
        )


def load_source_ops() -> SourceOps:
    from torch.utils.cpp_extension import load

    if not REGISTRATION_INCLUDE.is_dir():
        raise RuntimeError(f"missing kernel-builder registration include: {REGISTRATION_INCLUDE}")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _arch_list())
    namespace = "flashrt_vla_video_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "q_norm_rope_bf16.cu"),
        ],
        extra_include_paths=[str(PACKAGE / "csrc"), str(REGISTRATION_INCLUDE)],
        extra_cflags=["-O3", "-DCUDA_KERNEL"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-DCUDA_KERNEL"],
        verbose=False,
    )
    return SourceOps(namespace)


def load_installed_ops(artifact: str | None):
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return InstalledOps(importlib.import_module("flashrt_vla_video"))
    finally:
        if artifact:
            sys.path.remove(artifact)


def _ref_norm_rope(x, weight, cos, sin, eps=1e-6):
    half = x.shape[-1] // 2
    rstd = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)
    normed = x.float() * rstd * weight.float()
    lo = normed[..., :half]
    hi = normed[..., half:]
    out_lo = lo * cos.float() - hi * sin.float()
    out_hi = hi * cos.float() + lo * sin.float()
    return torch.cat([out_lo, out_hi], dim=-1).to(torch.bfloat16)


def _ref_qkv_split(packed, qw, kw, freqs_re, freqs_im, heads, head_dim, eps=1e-6):
    batch, tokens, _ = packed.shape
    dim = heads * head_dim
    q = packed[..., :dim].reshape(batch, tokens, heads, head_dim)
    k = packed[..., dim : 2 * dim].reshape(batch, tokens, heads, head_dim)
    qf = q.float()
    kf = k.float()
    qn = qf * torch.rsqrt((qf * qf).mean(dim=(-2, -1), keepdim=True) + eps)
    kn = kf * torch.rsqrt((kf * kf).mean(dim=(-2, -1), keepdim=True) + eps)
    qn = qn * qw.reshape(1, 1, heads, head_dim).float()
    kn = kn * kw.reshape(1, 1, heads, head_dim).float()

    def rope(x):
        xr = x[..., 0::2].float()
        xi = x[..., 1::2].float()
        fr = freqs_re[:tokens][None, :, None, :]
        fi = freqs_im[:tokens][None, :, None, :]
        out = torch.empty_like(x, dtype=torch.float32)
        out[..., 0::2] = xr * fr - xi * fi
        out[..., 1::2] = xr * fi + xi * fr
        return out.to(torch.bfloat16)

    return rope(qn), rope(kn)


def _check_q_norm(ops, shape) -> None:
    torch.manual_seed(0)
    q = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
    weight = (torch.randn(128, device="cuda", dtype=torch.bfloat16) * 0.1 + 1).contiguous()
    cos = torch.randn(64, device="cuda", dtype=torch.bfloat16).contiguous()
    sin = torch.randn(64, device="cuda", dtype=torch.bfloat16).contiguous()
    out = ops.q_norm_rope(q, weight, cos, sin)
    ref = _ref_norm_rope(q, weight, cos, sin)
    torch.testing.assert_close(out.float(), ref.float(), atol=0.03125, rtol=0)
    print(f"PASS q_norm_rope shape={shape}")


def _check_k_norm(ops, shape) -> None:
    torch.manual_seed(1)
    k = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
    v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
    weight = (torch.randn(128, device="cuda", dtype=torch.bfloat16) * 0.1 + 1).contiguous()
    cos = torch.randn(64, device="cuda", dtype=torch.bfloat16).contiguous()
    sin = torch.randn(64, device="cuda", dtype=torch.bfloat16).contiguous()
    k_out, v_out = ops.k_norm_rope_v_cache(k, v, weight, cos, sin)
    k_ref = _ref_norm_rope(k, weight, cos, sin)
    torch.testing.assert_close(k_out.float(), k_ref.float(), atol=0.03125, rtol=0)
    torch.testing.assert_close(v_out, v)
    print(f"PASS k_norm_rope_v_cache shape={shape}")


def _check_qkv_split(ops, tokens) -> None:
    torch.manual_seed(2)
    heads = 24
    head_dim = 128
    dim = heads * head_dim
    packed = (torch.randn((1, tokens, 3 * dim), device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
    qw = (torch.randn(dim, device="cuda", dtype=torch.bfloat16) * 0.1 + 1).contiguous()
    kw = (torch.randn(dim, device="cuda", dtype=torch.bfloat16) * 0.1 + 1).contiguous()
    freqs_re = torch.randn((128, head_dim // 2), device="cuda", dtype=torch.float32).contiguous()
    freqs_im = torch.randn((128, head_dim // 2), device="cuda", dtype=torch.float32).contiguous()
    q_out, k_out = ops.qkv_split_norm_rope(packed, qw, kw, freqs_re, freqs_im, heads=heads, head_dim=head_dim)
    q_ref, k_ref = _ref_qkv_split(packed, qw, kw, freqs_re, freqs_im, heads, head_dim)
    torch.testing.assert_close(q_out.float(), q_ref.float(), atol=0.03125, rtol=0)
    torch.testing.assert_close(k_out.float(), k_ref.float(), atol=0.03125, rtol=0)
    print(f"PASS qkv_split_norm_rope tokens={tokens}")


def run(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    shapes = [(1, 128), (8, 128)]
    if args.mode == "full":
        shapes += [(2, 4, 128)]
    for shape in shapes:
        _check_q_norm(ops, shape)
        _check_k_norm(ops, shape)
    for tokens in ([4, 64] if args.mode == "full" else [4]):
        _check_qkv_split(ops, tokens)
    print(f"PASS flashrt-vla-video {args.backend} mode={args.mode}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        import traceback
        traceback.print_exc()
        return 1
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps({"passed": 1, "total": 1, "backend": args.backend}) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
