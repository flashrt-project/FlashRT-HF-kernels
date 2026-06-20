#!/usr/bin/env python3
"""Correctness tests for linear-attention-primitives."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import importlib
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "linear-attention-primitives"
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

    def bf16_matvec(self, x, w):
        out = torch.empty((w.shape[0],), device=x.device, dtype=torch.bfloat16)
        self._ops.bf16_matvec(x, w, out)
        return out

    def bf16_smallm_matmul(self, x, w):
        out = torch.empty((x.shape[0], w.shape[0]), device=x.device, dtype=torch.bfloat16)
        self._ops.bf16_smallm_matmul(x, w, out)
        return out

    def split_qkv_broadcast_bf16(self, packed, q_heads, kv_heads, v_heads, head_dim):
        rows = packed.shape[0]
        q = torch.empty((rows, v_heads, head_dim), device=packed.device, dtype=torch.bfloat16)
        k = torch.empty_like(q)
        v = torch.empty_like(q)
        self._ops.split_qkv_broadcast_bf16(packed, q, k, v, q_heads, kv_heads, v_heads, head_dim)
        return q, k, v

    def partial_rope_qk_bf16(self, q_in, k_in, cos, sin, rope_dim):
        q_out = torch.empty_like(q_in)
        k_out = torch.empty_like(k_in)
        self._ops.partial_rope_qk_bf16(q_in, k_in, cos, sin, q_out, k_out, rope_dim)
        return q_out, k_out

    def gated_delta_prepare_bf16(self, a, b, neg_exp_a_log, dt_bias, heads=None, a_stride=None, b_stride=None):
        if heads is None:
            heads = neg_exp_a_log.shape[0]
        if a_stride is None:
            a_stride = a.shape[1]
        if b_stride is None:
            b_stride = b.shape[1]
        g = torch.empty((a.shape[0], heads), device=a.device, dtype=torch.bfloat16)
        beta = torch.empty_like(g)
        self._ops.gated_delta_prepare_bf16(a, b, neg_exp_a_log, dt_bias, g, beta, a_stride, b_stride)
        return g, beta


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
    namespace = "linear_attention_primitives_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "linear_attention_primitives.cu"),
        ],
        extra_include_paths=[str(PACKAGE / "csrc"), str(REGISTRATION_INCLUDE)],
        extra_cflags=["-O3", "-DCUDA_KERNEL"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-DCUDA_KERNEL"],
        is_python_module=False,
        verbose=False,
    )
    return SourceOps(namespace)


def load_installed_ops(artifact: str | None):
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return importlib.import_module("linear_attention_primitives")
    finally:
        if artifact:
            sys.path.remove(artifact)


def metrics(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float]:
    diff = (got.float() - ref.float()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    cos = torch.nn.functional.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0).item()
    return max_err, mean_err, cos


def assert_close(name: str, got: torch.Tensor, ref: torch.Tensor, max_atol: float, mean_atol: float, min_cos: float) -> None:
    max_err, mean_err, cos = metrics(got, ref)
    if max_err > max_atol or mean_err > mean_atol or cos < min_cos:
        raise AssertionError(
            f"{name}: max_err={max_err:.8f}, mean_err={mean_err:.8f}, cos={cos:.8f}"
        )


def rope_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dim: int) -> torch.Tensor:
    out = x.clone()
    half = rope_dim // 2
    left = x[:, :, :half].float()
    right = x[:, :, half:rope_dim].float()
    c_left = cos[:, None, :half].float()
    s_left = sin[:, None, :half].float()
    c_right = cos[:, None, half:rope_dim].float()
    s_right = sin[:, None, half:rope_dim].float()
    out[:, :, :half] = ((-right * s_left).to(torch.bfloat16).float() + left * c_left).to(torch.bfloat16)
    out[:, :, half:rope_dim] = ((left * s_right).to(torch.bfloat16).float() + right * c_right).to(torch.bfloat16)
    return out


def run_matmul_tests(ops) -> int:
    count = 0
    for k in [128, 4096, 5120]:
        for n in [512, 1024]:
            x = (torch.randn((k,), device="cuda") * 0.05).to(torch.bfloat16)
            w = (torch.randn((n, k), device="cuda") * 0.05).to(torch.bfloat16)
            got = ops.bf16_matvec(x, w)
            ref = (x.float() @ w.float().t()).to(torch.bfloat16)
            assert_close(f"bf16_matvec n={n} k={k}", got, ref, 0.03125, 0.0025, 0.9999)
            count += 1
    for m in [2, 3, 4]:
        k = 5120
        n = 96
        x2 = (torch.randn((m, k), device="cuda") * 0.05).to(torch.bfloat16)
        w2 = (torch.randn((n, k), device="cuda") * 0.05).to(torch.bfloat16)
        got = ops.bf16_smallm_matmul(x2, w2)
        ref = (x2.float() @ w2.float().t()).to(torch.bfloat16)
        assert_close(f"bf16_smallm_matmul m={m} n={n} k={k}", got, ref, 0.03125, 0.0025, 0.9999)
        count += 1
    return count


def run_layout_tests(ops) -> int:
    count = 0
    for rows, qh, kvh, vh, hd in [(1, 16, 16, 48, 128), (128, 16, 16, 48, 128)]:
        packed = torch.randn((rows, (qh + kvh + vh) * hd), device="cuda", dtype=torch.bfloat16)
        q, k, v = ops.split_qkv_broadcast_bf16(packed, qh, kvh, vh, hd)
        ref_v = packed[:, (qh + kvh) * hd :].reshape(rows, vh, hd).contiguous()
        ref_q_base = packed[:, : qh * hd].reshape(rows, qh, hd)
        ref_k_base = packed[:, qh * hd : (qh + kvh) * hd].reshape(rows, kvh, hd)
        q_idx = torch.arange(vh, device="cuda") * qh // vh
        k_idx = torch.arange(vh, device="cuda") * kvh // vh
        ref_q = ref_q_base[:, q_idx].contiguous()
        ref_k = ref_k_base[:, k_idx].contiguous()
        assert_close(f"split_qkv_broadcast_q rows={rows}", q, ref_q, 0.0, 0.0, 0.999999)
        assert_close(f"split_qkv_broadcast_k rows={rows}", k, ref_k, 0.0, 0.0, 0.999999)
        assert_close(f"split_qkv_broadcast_v rows={rows}", v, ref_v, 0.0, 0.0, 0.999999)
        count += 1
    return count


def run_rope_gating_tests(ops) -> int:
    count = 0
    for rows, qh, kh, hd, rd in [(1, 16, 16, 128, 64), (128, 24, 8, 128, 64), (256, 16, 16, 256, 128)]:
        q = torch.randn((rows, qh, hd), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((rows, kh, hd), device="cuda", dtype=torch.bfloat16)
        cos = torch.randn((rows, rd), device="cuda", dtype=torch.bfloat16)
        sin = torch.randn((rows, rd), device="cuda", dtype=torch.bfloat16)
        got_q, got_k = ops.partial_rope_qk_bf16(q, k, cos, sin, rd)
        assert_close(f"partial_rope_q rows={rows}", got_q, rope_ref(q, cos, sin, rd), 0.0, 0.0, 0.999999)
        assert_close(f"partial_rope_k rows={rows}", got_k, rope_ref(k, cos, sin, rd), 0.0, 0.0, 0.999999)
        count += 1

    for rows, heads, stride in [(1, 48, 48), (128, 48, 64), (1024, 48, 64)]:
        a = torch.randn((rows, stride), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((rows, stride), device="cuda", dtype=torch.bfloat16)
        neg = torch.randn((heads,), device="cuda", dtype=torch.float32) * 0.1
        bias = torch.randn((heads,), device="cuda", dtype=torch.float32) * 0.1
        got_g, got_beta = ops.gated_delta_prepare_bf16(a, b, neg, bias, heads=heads, a_stride=stride, b_stride=stride)
        ref_g = (neg[None, :] * torch.nn.functional.softplus(a[:, :heads].float() + bias[None, :])).to(torch.bfloat16)
        ref_beta = torch.sigmoid(b[:, :heads].float()).to(torch.bfloat16)
        assert_close(f"gated_delta_g rows={rows}", got_g, ref_g, 0.0, 0.0, 0.999999)
        assert_close(f"gated_delta_beta rows={rows}", got_beta, ref_beta, 0.0, 0.0, 0.999999)
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(0)
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    total = run_matmul_tests(ops) + run_layout_tests(ops) + run_rope_gating_tests(ops)
    torch.cuda.synchronize()
    print(f"linear-attention-primitives correctness passed: {total} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
