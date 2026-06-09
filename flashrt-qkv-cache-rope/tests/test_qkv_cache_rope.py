#!/usr/bin/env python3
"""Correctness tests for flashrt-qkv-cache-rope."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import importlib
import math
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "flashrt-qkv-cache-rope"
REGISTRATION_INCLUDE = (
    ROOT.parent
    / "kernels"
    / "kernel-builder"
    / "src"
    / "pyproject"
    / "templates"
    / "torch"
)


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self._ops = getattr(torch.ops, namespace)

    def decode_q_norm_rope_stage_bf16(
        self, q_pre, q_norm_weight, cos, sin, eps=1e-6, q_out=None
    ):
        if q_out is None:
            q_out = torch.empty_like(q_pre)
        self._ops.decode_q_norm_rope_stage_bf16(
            q_pre, q_norm_weight, cos, sin, float(eps), q_out
        )
        return q_out

    def decode_k_norm_rope_kvwrite_bf16(
        self, k_pre, v_pre, k_norm_weight, cos, sin, eps=1e-6, k_cache_dst=None, v_cache_dst=None
    ):
        if k_cache_dst is None:
            k_cache_dst = torch.empty_like(k_pre)
        if v_cache_dst is None:
            v_cache_dst = torch.empty_like(v_pre)
        self._ops.decode_k_norm_rope_kvwrite_bf16(
            k_pre, v_pre, k_norm_weight, cos, sin, float(eps), k_cache_dst, v_cache_dst
        )
        return k_cache_dst, v_cache_dst

    def decode_k_norm_rope_kvwrite_devpos_bf16(
        self, k_pre, v_pre, k_norm_weight, cos, sin, cur_pos, k_cache, v_cache, eps=1e-6
    ):
        self._ops.decode_k_norm_rope_kvwrite_devpos_bf16(
            k_pre, v_pre, k_norm_weight, cos, sin, cur_pos, float(eps), k_cache, v_cache
        )
        return k_cache, v_cache

    def qkv_split_rope_kvcache_bf16(
        self,
        packed_qkv,
        rope,
        q_heads,
        kv_heads,
        head_dim,
        cache_offset,
        q_out=None,
        k_cache=None,
        v_cache=None,
        max_seq_len=None,
    ):
        batch, seq_len, _ = packed_qkv.shape
        if q_out is None:
            q_out = torch.empty(
                (batch, seq_len, q_heads, head_dim),
                device=packed_qkv.device,
                dtype=torch.bfloat16,
            )
        if k_cache is None or v_cache is None:
            if max_seq_len is None:
                max_seq_len = cache_offset + seq_len
            shape = (batch, max_seq_len, kv_heads, head_dim)
            if k_cache is None:
                k_cache = torch.empty(shape, device=packed_qkv.device, dtype=torch.bfloat16)
            if v_cache is None:
                v_cache = torch.empty(shape, device=packed_qkv.device, dtype=torch.bfloat16)
        self._ops.qkv_split_rope_kvcache_bf16(
            packed_qkv,
            rope,
            int(q_heads),
            int(kv_heads),
            int(head_dim),
            int(cache_offset),
            q_out,
            k_cache,
            v_cache,
        )
        return q_out, k_cache, v_cache

    def qkv_split_bf16(
        self,
        packed_qkv,
        heads,
        head_dim,
        q_out=None,
        k_out=None,
        v_out=None,
    ):
        out_shape = (packed_qkv.shape[0], packed_qkv.shape[1], heads, head_dim)
        if q_out is None:
            q_out = torch.empty(out_shape, device=packed_qkv.device, dtype=torch.bfloat16)
        if k_out is None:
            k_out = torch.empty_like(q_out)
        if v_out is None:
            v_out = torch.empty_like(q_out)
        self._ops.qkv_split_bf16(
            packed_qkv,
            int(heads),
            int(head_dim),
            q_out,
            k_out,
            v_out,
        )
        return q_out, k_out, v_out

    def qkv_split_norm_rope_bf16(
        self,
        packed_qkv,
        norm_q_weight,
        norm_k_weight,
        freqs_re,
        freqs_im,
        heads,
        head_dim,
        rope_seq_len=None,
        eps=1e-6,
        q_out=None,
        k_out=None,
    ):
        if rope_seq_len is None:
            rope_seq_len = packed_qkv.shape[1]
        if q_out is None:
            q_out = torch.empty(
                (packed_qkv.shape[0], packed_qkv.shape[1], heads, head_dim),
                device=packed_qkv.device,
                dtype=torch.bfloat16,
            )
        if k_out is None:
            k_out = torch.empty_like(q_out)
        self._ops.qkv_split_norm_rope_bf16(
            packed_qkv,
            norm_q_weight,
            norm_k_weight,
            freqs_re,
            freqs_im,
            int(heads),
            int(head_dim),
            int(rope_seq_len),
            float(eps),
            q_out,
            k_out,
        )
        return q_out, k_out

    def qkv_split_bias_norm_rope_v_bf16(
        self,
        packed_qkv,
        qkv_bias,
        norm_q_weight,
        norm_k_weight,
        freqs_re,
        freqs_im,
        heads,
        head_dim,
        rope_seq_len=None,
        eps=1e-6,
        q_out=None,
        k_out=None,
        v_out=None,
    ):
        if rope_seq_len is None:
            rope_seq_len = packed_qkv.shape[1]
        out_shape = (packed_qkv.shape[0], packed_qkv.shape[1], heads, head_dim)
        if q_out is None:
            q_out = torch.empty(out_shape, device=packed_qkv.device, dtype=torch.bfloat16)
        if k_out is None:
            k_out = torch.empty_like(q_out)
        if v_out is None:
            v_out = torch.empty_like(q_out)
        self._ops.qkv_split_bias_norm_rope_v_bf16(
            packed_qkv,
            qkv_bias,
            norm_q_weight,
            norm_k_weight,
            freqs_re,
            freqs_im,
            int(heads),
            int(head_dim),
            int(rope_seq_len),
            float(eps),
            q_out,
            k_out,
            v_out,
        )
        return q_out, k_out, v_out

    def qkv_split_bias_norm_rope_v_cat_bf16(
        self,
        packed_qkv,
        qkv_bias,
        norm_q_weight,
        norm_k_weight,
        freqs_re,
        freqs_im,
        heads,
        head_dim,
        video_offset,
        q_cat_out,
        k_cat_out,
        v_cat_out,
        rope_seq_len=None,
        eps=1e-6,
    ):
        if rope_seq_len is None:
            rope_seq_len = packed_qkv.shape[1]
        self._ops.qkv_split_bias_norm_rope_v_cat_bf16(
            packed_qkv,
            qkv_bias,
            norm_q_weight,
            norm_k_weight,
            freqs_re,
            freqs_im,
            int(heads),
            int(head_dim),
            int(video_offset),
            int(rope_seq_len),
            float(eps),
            q_cat_out,
            k_cat_out,
            v_cat_out,
        )
        return q_cat_out, k_cat_out, v_cat_out

    def qkv_split_joint3_cat_bf16(
        self,
        packed_v,
        qkv_v_bias,
        norm_v_q_weight,
        norm_v_k_weight,
        freqs_re,
        freqs_im,
        packed_a,
        norm_a_q_weight,
        norm_a_k_weight,
        packed_u,
        norm_u_q_weight,
        norm_u_k_weight,
        heads,
        head_dim,
        q_cat_out,
        k_cat_out,
        v_cat_out,
        rope_seq_len=None,
        eps_v=1e-6,
        eps_a=1e-6,
        eps_u=1e-6,
    ):
        if rope_seq_len is None:
            rope_seq_len = packed_v.shape[1]
        self._ops.qkv_split_joint3_cat_bf16(
            packed_v,
            qkv_v_bias,
            norm_v_q_weight,
            norm_v_k_weight,
            freqs_re,
            freqs_im,
            packed_a,
            norm_a_q_weight,
            norm_a_k_weight,
            packed_u,
            norm_u_q_weight,
            norm_u_k_weight,
            int(heads),
            int(head_dim),
            int(rope_seq_len),
            float(eps_v),
            float(eps_a),
            float(eps_u),
            q_cat_out,
            k_cat_out,
            v_cat_out,
        )
        return q_cat_out, k_cat_out, v_cat_out


def _preload_cublaslt() -> None:
    for parent in Path(torch.__file__).resolve().parents:
        candidate = parent / "nvidia" / "cublas" / "lib" / "libcublasLt.so.12"
        if candidate.exists():
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
            return
    library = ctypes.util.find_library("cublasLt")
    if library:
        ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)


def _current_arch_list() -> str:
    major, minor = torch.cuda.get_device_capability(0)
    return f"{major}.{minor}"


def load_source_ops() -> SourceOps:
    from torch.utils.cpp_extension import load

    if not REGISTRATION_INCLUDE.is_dir():
        raise RuntimeError(f"missing kernel-builder registration include: {REGISTRATION_INCLUDE}")
    _preload_cublaslt()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _current_arch_list())
    namespace = "flashrt_qkv_cache_rope_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "qkv_cache_rope.cu"),
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
        return importlib.import_module("flashrt_qkv_cache_rope")
    finally:
        if artifact:
            sys.path.remove(artifact)


def make_freqs(seq_len: int, head_dim: int):
    theta = torch.randn((seq_len, head_dim // 2), device="cuda", dtype=torch.float32)
    return torch.cos(theta).contiguous(), torch.sin(theta).contiguous()


def make_decode_case(heads: int):
    q = torch.randn((heads, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((heads, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn((heads, 128), device="cuda", dtype=torch.bfloat16)
    q_w = (1.0 + 0.1 * torch.randn((128,), device="cuda", dtype=torch.bfloat16)).contiguous()
    k_w = (1.0 + 0.1 * torch.randn((128,), device="cuda", dtype=torch.bfloat16)).contiguous()
    theta = torch.randn((64,), device="cuda", dtype=torch.float32)
    cos = torch.cos(theta).to(torch.bfloat16).contiguous()
    sin = torch.sin(theta).to(torch.bfloat16).contiguous()
    return q, k, v, q_w, k_w, cos, sin


def make_interleaved_rope(seq_len: int, head_dim: int) -> torch.Tensor:
    theta = torch.randn((seq_len, head_dim // 2), device="cuda", dtype=torch.float32)
    cos = torch.cos(theta).to(torch.bfloat16)
    sin = torch.sin(theta).to(torch.bfloat16)
    return torch.stack([cos, sin], dim=-1).reshape(seq_len, head_dim).contiguous()


def make_case(batch: int, seq_len: int, heads: int, head_dim: int):
    dim = heads * head_dim
    packed_qkv = torch.randn((batch, seq_len, 3 * dim), device="cuda", dtype=torch.bfloat16)
    norm_q_weight = (1.0 + 0.1 * torch.randn((dim,), device="cuda", dtype=torch.bfloat16)).contiguous()
    norm_k_weight = (1.0 + 0.1 * torch.randn((dim,), device="cuda", dtype=torch.bfloat16)).contiguous()
    freqs_re, freqs_im = make_freqs(seq_len, head_dim)
    return packed_qkv, norm_q_weight, norm_k_weight, freqs_re, freqs_im


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    rms = torch.rsqrt(torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + eps)
    return x.float() * rms * weight.float()


def apply_pair_rope(x: torch.Tensor, freqs_re: torch.Tensor, freqs_im: torch.Tensor, rope_seq_len: int):
    # x: (B, L, H, D), pairs are adjacent even/odd elements within each head.
    y = x.clone()
    if rope_seq_len <= 0:
        return y
    x_rope = x[:, :rope_seq_len].float().reshape(
        x.shape[0], rope_seq_len, x.shape[2], x.shape[3] // 2, 2
    )
    re = x_rope[..., 0]
    im = x_rope[..., 1]
    fr = freqs_re[:rope_seq_len].view(1, rope_seq_len, 1, x.shape[3] // 2)
    fi = freqs_im[:rope_seq_len].view(1, rope_seq_len, 1, x.shape[3] // 2)
    out = torch.empty_like(x_rope.float())
    out[..., 0] = re * fr - im * fi
    out[..., 1] = re * fi + im * fr
    y[:, :rope_seq_len] = out.reshape(
        x.shape[0], rope_seq_len, x.shape[2], x.shape[3]
    ).to(torch.bfloat16)
    return y


def apply_interleaved_pair_rope(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    # x: (B, L, H, D), rope: (L, D) as [cos0, sin0, cos1, sin1, ...].
    batch, seq_len, heads, head_dim = x.shape
    xf = x.float().reshape(batch, seq_len, heads, head_dim // 2, 2)
    re = xf[..., 0]
    im = xf[..., 1]
    rope_f = rope[:seq_len].float().reshape(seq_len, head_dim // 2, 2)
    cos = rope_f[..., 0].view(1, seq_len, 1, head_dim // 2)
    sin = rope_f[..., 1].view(1, seq_len, 1, head_dim // 2)
    out = torch.empty_like(xf)
    out[..., 0] = re * cos - im * sin
    out[..., 1] = re * sin + im * cos
    return out.reshape(batch, seq_len, heads, head_dim).to(torch.bfloat16)


def apply_rotate_half_rope_128(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    half = 64
    out = torch.empty_like(xf)
    c = cos.float().view(1, half)
    s = sin.float().view(1, half)
    out[:, :half] = xf[:, :half] * c - xf[:, half:] * s
    out[:, half:] = xf[:, half:] * c + xf[:, :half] * s
    return out.to(torch.bfloat16)


def ref_qkv_split_norm_rope(packed_qkv, norm_q_weight, norm_k_weight, freqs_re, freqs_im, heads, head_dim, rope_seq_len, eps):
    batch, seq_len, _ = packed_qkv.shape
    dim = heads * head_dim
    q = packed_qkv[:, :, :dim]
    k = packed_qkv[:, :, dim : 2 * dim]
    qn = rms_norm(q, norm_q_weight, eps).to(torch.bfloat16).view(batch, seq_len, heads, head_dim)
    kn = rms_norm(k, norm_k_weight, eps).to(torch.bfloat16).view(batch, seq_len, heads, head_dim)
    return (
        apply_pair_rope(qn, freqs_re, freqs_im, rope_seq_len),
        apply_pair_rope(kn, freqs_re, freqs_im, rope_seq_len),
    )


def ref_qkv_split_bias_norm_rope_v(
    packed_qkv,
    qkv_bias,
    norm_q_weight,
    norm_k_weight,
    freqs_re,
    freqs_im,
    heads,
    head_dim,
    rope_seq_len,
    eps,
):
    batch, seq_len, _ = packed_qkv.shape
    dim = heads * head_dim
    biased = packed_qkv.float() + qkv_bias.float().view(1, 1, 3 * dim)
    q = biased[:, :, :dim]
    k = biased[:, :, dim : 2 * dim]
    v = biased[:, :, 2 * dim :].to(torch.bfloat16).view(batch, seq_len, heads, head_dim)
    qn = rms_norm(q, norm_q_weight, eps).to(torch.bfloat16).view(batch, seq_len, heads, head_dim)
    kn = rms_norm(k, norm_k_weight, eps).to(torch.bfloat16).view(batch, seq_len, heads, head_dim)
    return (
        apply_pair_rope(qn, freqs_re, freqs_im, rope_seq_len),
        apply_pair_rope(kn, freqs_re, freqs_im, rope_seq_len),
        v,
    )


def ref_norm_qkv_no_rope(packed_qkv, norm_q_weight, norm_k_weight, heads, head_dim, eps):
    batch, seq_len, _ = packed_qkv.shape
    dim = heads * head_dim
    q = packed_qkv[:, :, :dim]
    k = packed_qkv[:, :, dim : 2 * dim]
    v = packed_qkv[:, :, 2 * dim :]
    qn = rms_norm(q, norm_q_weight, eps).to(torch.bfloat16).view(batch, seq_len, heads, head_dim)
    kn = rms_norm(k, norm_k_weight, eps).to(torch.bfloat16).view(batch, seq_len, heads, head_dim)
    return qn, kn, v.view(batch, seq_len, heads, head_dim)


def ref_decode_norm_rope(x: torch.Tensor, weight: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, eps: float):
    normed = rms_norm(x, weight, eps).to(torch.bfloat16)
    return apply_rotate_half_rope_128(normed, cos, sin)


def ref_qkv_split_rope_kvcache(packed_qkv, rope, q_heads, kv_heads, head_dim):
    batch, seq_len, _ = packed_qkv.shape
    q_dim = q_heads * head_dim
    kv_dim = kv_heads * head_dim
    q = packed_qkv[:, :, :q_dim].view(batch, seq_len, q_heads, head_dim)
    k = packed_qkv[:, :, q_dim : q_dim + kv_dim].view(batch, seq_len, kv_heads, head_dim)
    v = packed_qkv[:, :, q_dim + kv_dim :].view(batch, seq_len, kv_heads, head_dim)
    return (
        apply_interleaved_pair_rope(q, rope),
        apply_interleaved_pair_rope(k, rope),
        v,
    )


def percentile(x: torch.Tensor, q: float) -> torch.Tensor:
    flat = x.flatten()
    k = max(1, min(flat.numel(), math.ceil(q * flat.numel())))
    return flat.kthvalue(k).values


def assert_close_distribution(name: str, got: torch.Tensor, expected: torch.Tensor) -> None:
    diff = (got.float() - expected.float()).abs().flatten()
    rel = diff / expected.float().abs().flatten().clamp_min(1.0)
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    p99_abs = float(percentile(diff, 0.99).item())
    p99_rel = float(percentile(rel, 0.99).item())
    cosine = float(torch.nn.functional.cosine_similarity(got.float().flatten(), expected.float().flatten(), dim=0).item())
    if p99_abs > 0.015625 or p99_rel > 0.02:
        raise AssertionError(
            f"{name} failed: max_abs={max_abs} mean_abs={mean_abs} "
            f"p99_abs={p99_abs} p99_rel={p99_rel} cosine={cosine}"
        )
    print(
        f"PASS {name}: max_abs={max_abs:.6f} mean_abs={mean_abs:.6f} "
        f"p99_abs={p99_abs:.6f} p99_rel={p99_rel:.6f} cosine={cosine:.8f}"
    )


def expect_runtime_error(label: str, fn) -> None:
    try:
        fn()
    except RuntimeError as exc:
        print(f"PASS {label}: rejected invalid input ({str(exc).splitlines()[0]})")
        return
    raise AssertionError(f"{label} failed: expected RuntimeError")


def run_shape(ops, label: str, shape: tuple[int, int, int, int], eps: float) -> None:
    batch, seq_len, heads, head_dim = shape
    packed, q_w, k_w, freqs_re, freqs_im = make_case(batch, seq_len, heads, head_dim)
    rope_seq_len = seq_len
    got_q, got_k = ops.qkv_split_norm_rope_bf16(
        packed, q_w, k_w, freqs_re, freqs_im, heads, head_dim, rope_seq_len, eps
    )
    exp_q, exp_k = ref_qkv_split_norm_rope(
        packed, q_w, k_w, freqs_re, freqs_im, heads, head_dim, rope_seq_len, eps
    )
    assert_close_distribution(f"{label}/q", got_q, exp_q)
    assert_close_distribution(f"{label}/k", got_k, exp_k)

    if seq_len > 1:
        got_q2, got_k2 = ops.qkv_split_norm_rope_bf16(
            packed, q_w, k_w, freqs_re, freqs_im, heads, head_dim, seq_len - 1, eps
        )
        exp_q2, exp_k2 = ref_qkv_split_norm_rope(
            packed, q_w, k_w, freqs_re, freqs_im, heads, head_dim, seq_len - 1, eps
        )
        assert_close_distribution(f"{label}/partial_rope_q", got_q2, exp_q2)
        assert_close_distribution(f"{label}/partial_rope_k", got_k2, exp_k2)


def run_plain_split_shape(ops, label: str, shape: tuple[int, int, int, int]) -> None:
    batch, seq_len, heads, head_dim = shape
    dim = heads * head_dim
    packed = torch.randn((batch, seq_len, 3 * dim), device="cuda", dtype=torch.bfloat16)
    got_q, got_k, got_v = ops.qkv_split_bf16(packed, heads, head_dim)
    exp_q = packed[:, :, :dim].view(batch, seq_len, heads, head_dim).contiguous()
    exp_k = packed[:, :, dim : 2 * dim].view(batch, seq_len, heads, head_dim).contiguous()
    exp_v = packed[:, :, 2 * dim :].view(batch, seq_len, heads, head_dim).contiguous()
    assert_close_distribution(f"{label}/plain_q", got_q, exp_q)
    assert_close_distribution(f"{label}/plain_k", got_k, exp_k)
    assert_close_distribution(f"{label}/plain_v", got_v, exp_v)


def run_bias_and_cat_shape(ops, label: str, shape: tuple[int, int, int, int], eps: float) -> None:
    batch, seq_len, heads, head_dim = shape
    dim = heads * head_dim
    packed, q_w, k_w, freqs_re, freqs_im = make_case(batch, seq_len, heads, head_dim)
    qkv_bias = (0.02 * torch.randn((3 * dim,), device="cuda", dtype=torch.bfloat16)).contiguous()
    got_q, got_k, got_v = ops.qkv_split_bias_norm_rope_v_bf16(
        packed, qkv_bias, q_w, k_w, freqs_re, freqs_im, heads, head_dim, seq_len, eps
    )
    exp_q, exp_k, exp_v = ref_qkv_split_bias_norm_rope_v(
        packed, qkv_bias, q_w, k_w, freqs_re, freqs_im, heads, head_dim, seq_len, eps
    )
    assert_close_distribution(f"{label}/bias_v_q", got_q, exp_q)
    assert_close_distribution(f"{label}/bias_v_k", got_k, exp_k)
    assert_close_distribution(f"{label}/bias_v_v", got_v, exp_v)

    video_offset = 3
    total_seq_len = video_offset + seq_len + 5
    q_cat = torch.full((batch, total_seq_len, heads, head_dim), -7.0, device="cuda", dtype=torch.bfloat16)
    k_cat = torch.full_like(q_cat, -8.0)
    v_cat = torch.full_like(q_cat, -9.0)
    ops.qkv_split_bias_norm_rope_v_cat_bf16(
        packed,
        qkv_bias,
        q_w,
        k_w,
        freqs_re,
        freqs_im,
        heads,
        head_dim,
        video_offset,
        q_cat,
        k_cat,
        v_cat,
        seq_len,
        eps,
    )
    sl = slice(video_offset, video_offset + seq_len)
    assert_close_distribution(f"{label}/bias_cat_q", q_cat[:, sl], exp_q)
    assert_close_distribution(f"{label}/bias_cat_k", k_cat[:, sl], exp_k)
    assert_close_distribution(f"{label}/bias_cat_v", v_cat[:, sl], exp_v)
    if float(q_cat[:, :video_offset].float().mean().item()) != -7.0:
        raise AssertionError(f"{label}/bias_cat_prefix failed: q prefix was modified")


def run_joint3_shape(ops, label: str, heads: int, head_dim: int, eps: float) -> None:
    batch = 1
    l_v, l_a, l_u = (64, 8, 4) if label == "small" else (256, 16, 16)
    packed_v, v_q_w, v_k_w, freqs_re, freqs_im = make_case(batch, l_v, heads, head_dim)
    packed_a, a_q_w, a_k_w, _, _ = make_case(batch, l_a, heads, head_dim)
    packed_u, u_q_w, u_k_w, _, _ = make_case(batch, l_u, heads, head_dim)
    dim = heads * head_dim
    qkv_v_bias = (0.02 * torch.randn((3 * dim,), device="cuda", dtype=torch.bfloat16)).contiguous()
    total = l_v + l_a + l_u
    q_cat = torch.empty((batch, total, heads, head_dim), device="cuda", dtype=torch.bfloat16)
    k_cat = torch.empty_like(q_cat)
    v_cat = torch.empty_like(q_cat)
    ops.qkv_split_joint3_cat_bf16(
        packed_v,
        qkv_v_bias,
        v_q_w,
        v_k_w,
        freqs_re,
        freqs_im,
        packed_a,
        a_q_w,
        a_k_w,
        packed_u,
        u_q_w,
        u_k_w,
        heads,
        head_dim,
        q_cat,
        k_cat,
        v_cat,
        l_v,
        eps,
        eps,
        eps,
    )
    exp_vq, exp_vk, exp_vv = ref_qkv_split_bias_norm_rope_v(
        packed_v, qkv_v_bias, v_q_w, v_k_w, freqs_re, freqs_im, heads, head_dim, l_v, eps
    )
    exp_aq, exp_ak, exp_av = ref_norm_qkv_no_rope(packed_a, a_q_w, a_k_w, heads, head_dim, eps)
    exp_uq, exp_uk, exp_uv = ref_norm_qkv_no_rope(packed_u, u_q_w, u_k_w, heads, head_dim, eps)
    exp_q = torch.cat([exp_vq, exp_aq, exp_uq], dim=1)
    exp_k = torch.cat([exp_vk, exp_ak, exp_uk], dim=1)
    exp_v = torch.cat([exp_vv, exp_av, exp_uv], dim=1)
    assert_close_distribution(f"{label}/joint3_q", q_cat, exp_q)
    assert_close_distribution(f"{label}/joint3_k", k_cat, exp_k)
    assert_close_distribution(f"{label}/joint3_v", v_cat, exp_v)


def run_decode_shape(ops, label: str, heads: int, eps: float) -> None:
    q, k, v, q_w, k_w, cos, sin = make_decode_case(heads)
    got_q = ops.decode_q_norm_rope_stage_bf16(q, q_w, cos, sin, eps)
    exp_q = ref_decode_norm_rope(q, q_w, cos, sin, eps)
    assert_close_distribution(f"{label}/decode_q_stage", got_q, exp_q)

    got_k, got_v = ops.decode_k_norm_rope_kvwrite_bf16(k, v, k_w, cos, sin, eps)
    exp_k = ref_decode_norm_rope(k, k_w, cos, sin, eps)
    assert_close_distribution(f"{label}/decode_k_cache", got_k, exp_k)
    assert_close_distribution(f"{label}/decode_v_cache", got_v, v)

    max_seq = 8
    pos = 3
    k_cache = torch.full((max_seq, heads, 128), -7.0, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.full((max_seq, heads, 128), -9.0, device="cuda", dtype=torch.bfloat16)
    cur_pos = torch.tensor([pos], device="cuda", dtype=torch.int32)
    ops.decode_k_norm_rope_kvwrite_devpos_bf16(k, v, k_w, cos, sin, cur_pos, k_cache, v_cache, eps)
    assert_close_distribution(f"{label}/decode_devpos_k", k_cache[pos], exp_k)
    assert_close_distribution(f"{label}/decode_devpos_v", v_cache[pos], v)
    if float(k_cache[:pos].float().mean().item()) != -7.0:
        raise AssertionError(f"{label}/decode_devpos_k_prefix failed: prefix modified")
    if float(v_cache[pos + 1 :].float().mean().item()) != -9.0:
        raise AssertionError(f"{label}/decode_devpos_v_suffix failed: suffix modified")


def run_kvcache_shape(
    ops,
    label: str,
    batch: int,
    seq_len: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> None:
    qkv_dim = (q_heads + 2 * kv_heads) * head_dim
    packed = torch.randn((batch, seq_len, qkv_dim), device="cuda", dtype=torch.bfloat16)
    rope = make_interleaved_rope(seq_len, head_dim)
    max_seq_len = seq_len + 5
    cache_offset = 2
    q_out = torch.empty((batch, seq_len, q_heads, head_dim), device="cuda", dtype=torch.bfloat16)
    k_cache = torch.full(
        (batch, max_seq_len, kv_heads, head_dim), -7.0, device="cuda", dtype=torch.bfloat16
    )
    v_cache = torch.full_like(k_cache, -9.0)
    ops.qkv_split_rope_kvcache_bf16(
        packed,
        rope,
        q_heads,
        kv_heads,
        head_dim,
        cache_offset,
        q_out,
        k_cache,
        v_cache,
    )
    exp_q, exp_k, exp_v = ref_qkv_split_rope_kvcache(
        packed, rope, q_heads, kv_heads, head_dim
    )
    sl = slice(cache_offset, cache_offset + seq_len)
    assert_close_distribution(f"{label}/kvcache_q", q_out, exp_q)
    assert_close_distribution(f"{label}/kvcache_k", k_cache[:, sl], exp_k)
    assert_close_distribution(f"{label}/kvcache_v", v_cache[:, sl], exp_v)
    if float(k_cache[:, :cache_offset].float().mean().item()) != -7.0:
        raise AssertionError(f"{label}/kvcache_k_prefix failed: prefix modified")
    if float(v_cache[:, cache_offset + seq_len :].float().mean().item()) != -9.0:
        raise AssertionError(f"{label}/kvcache_v_suffix failed: suffix modified")


def run_rejection_tests(ops) -> None:
    packed, q_w, k_w, freqs_re, freqs_im = make_case(1, 4, 4, 128)
    expect_runtime_error(
        "reject bad packed cols",
        lambda: ops.qkv_split_norm_rope_bf16(
            packed[:, :, :-2].contiguous(), q_w, k_w, freqs_re, freqs_im, 4, 128
        ),
    )
    expect_runtime_error(
        "reject odd head_dim",
        lambda: ops.qkv_split_norm_rope_bf16(
            packed, q_w, k_w, freqs_re, freqs_im, 4, 127
        ),
    )
    q, _, _, q_w, _, cos, sin = make_decode_case(4)
    expect_runtime_error(
        "reject decode head_dim",
        lambda: ops.decode_q_norm_rope_stage_bf16(
            q[:, :-1].contiguous(), q_w, cos, sin
        ),
    )
    packed_gqa = torch.randn((1, 10, (8 + 2 * 1) * 256), device="cuda", dtype=torch.bfloat16)
    rope = make_interleaved_rope(10, 256)
    q_out = torch.empty((1, 10, 8, 256), device="cuda", dtype=torch.bfloat16)
    k_cache = torch.empty((1, 12, 1, 256), device="cuda", dtype=torch.bfloat16)
    v_cache = torch.empty_like(k_cache)
    expect_runtime_error(
        "reject kvcache bounds",
        lambda: ops.qkv_split_rope_kvcache_bf16(
            packed_gqa, rope, 8, 1, 256, 4, q_out, k_cache, v_cache
        ),
    )
    expect_runtime_error(
        "reject kvcache packed cols",
        lambda: ops.qkv_split_rope_kvcache_bf16(
            packed_gqa[:, :, :-1].contiguous(), rope, 8, 1, 256, 0, q_out, k_cache, v_cache
        ),
    )
    expect_runtime_error(
        "reject plain split output shape",
        lambda: ops.qkv_split_bf16(
            torch.randn((1, 4, 3 * 4 * 128), device="cuda", dtype=torch.bfloat16),
            4,
            128,
            torch.empty((1, 4, 4, 64), device="cuda", dtype=torch.bfloat16),
            torch.empty((1, 4, 4, 128), device="cuda", dtype=torch.bfloat16),
            torch.empty((1, 4, 4, 128), device="cuda", dtype=torch.bfloat16),
        ),
    )


def run(args) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(31)
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    shapes = {
        "small": (1, 4, 4, 128),
        "wan_1k": (1, 1024, 24, 128),
        "wan_2520": (1, 2520, 24, 128),
        "vl_512": (1, 512, 16, 128),
    }
    if args.mode == "smoke":
        shapes = {k: shapes[k] for k in ("small",)}
    for label, shape in shapes.items():
        run_plain_split_shape(ops, label, shape)
        run_shape(ops, label, shape, args.eps)
        run_bias_and_cat_shape(ops, label, shape, args.eps)
    run_plain_split_shape(ops, "siglip_plain", (2, 256, 16, 72))
    run_decode_shape(ops, "decode_small", 4, args.eps)
    run_kvcache_shape(ops, "pi05_decoder_gqa", 1, 10, 8, 1, 256)
    if args.mode == "full":
        run_decode_shape(ops, "decode_vla", 24, args.eps)
        run_kvcache_shape(ops, "gqa_batch2", 2, 16, 8, 2, 128)
    run_joint3_shape(ops, "small", 4, 128, args.eps)
    if args.mode == "full":
        run_joint3_shape(ops, "vla", 24, 128, args.eps)
    run_rejection_tests(ops)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
