#!/usr/bin/env python3
"""Correctness tests for sageattention2-blackwell."""

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
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "sageattention2-blackwell"
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

    @staticmethod
    def padded_k64(seqlen_k: int) -> int:
        return ((int(seqlen_k) + 63) // 64) * 64

    @staticmethod
    def q_scale_elems(batch: int, seqlen_q: int, q_heads: int) -> int:
        return int(batch) * int(q_heads) * ((int(seqlen_q) + 31) // 32)

    @staticmethod
    def k_scale_elems(batch: int, seqlen_k: int, kv_heads: int) -> int:
        return int(batch) * int(kv_heads) * ((int(seqlen_k) + 63) // 64)

    @staticmethod
    def v_scale_elems(batch: int, kv_heads: int) -> int:
        return int(batch) * int(kv_heads) * 128

    def quantize_q_bf16_d128(self, q, q_i8=None, q_scale=None):
        q_i8 = torch.empty_like(q, dtype=torch.int8) if q_i8 is None else q_i8
        if q_scale is None:
            q_scale = torch.empty((self.q_scale_elems(q.shape[0], q.shape[1], q.shape[2]),), device=q.device, dtype=torch.float32)
        self._ops.quantize_q_bf16_d128(q, q_i8, q_scale)
        return q_i8, q_scale

    def quantize_k_bf16_d128(self, k, k_i8=None, k_scale=None):
        k_i8 = torch.empty_like(k, dtype=torch.int8) if k_i8 is None else k_i8
        if k_scale is None:
            k_scale = torch.empty((self.k_scale_elems(k.shape[0], k.shape[1], k.shape[2]),), device=k.device, dtype=torch.float32)
        self._ops.quantize_k_bf16_d128(k, k_i8, k_scale)
        return k_i8, k_scale

    def quantize_v_fp16_bf16_d128(self, v, v_half=None):
        v_half = torch.empty_like(v, dtype=torch.float16) if v_half is None else v_half
        self._ops.quantize_v_fp16_bf16_d128(v, v_half)
        return v_half

    def quantize_v_fp8_bf16_d128(self, v, v_fp8_tpp=None, v_scale=None):
        if v_fp8_tpp is None:
            v_fp8_tpp = torch.empty((v.shape[0], 128, v.shape[2], self.padded_k64(v.shape[1])), device=v.device, dtype=torch.int8)
        if v_scale is None:
            v_scale = torch.empty((self.v_scale_elems(v.shape[0], v.shape[2]),), device=v.device, dtype=torch.float32)
        self._ops.quantize_v_fp8_bf16_d128(v, v_fp8_tpp, v_scale)
        return v_fp8_tpp, v_scale

    def sage2_qk_int8_sv_f16_bf16_d128(self, q_i8, k_i8, v_half, q_scale, k_scale, *, softmax_scale=None, causal=False, out=None):
        out = torch.empty_like(q_i8, dtype=torch.bfloat16) if out is None else out
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(128)
        self._ops.sage2_qk_int8_sv_f16_bf16_d128(q_i8, k_i8, v_half, q_scale, k_scale, float(softmax_scale), bool(causal), out)
        return out

    def sage2_qk_int8_sv_f8_bf16_d128(self, q_i8, k_i8, v_fp8_tpp, q_scale, k_scale, v_scale, *, softmax_scale=None, causal=False, out=None):
        out = torch.empty_like(q_i8, dtype=torch.bfloat16) if out is None else out
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(128)
        self._ops.sage2_qk_int8_sv_f8_bf16_d128(q_i8, k_i8, v_fp8_tpp, q_scale, k_scale, v_scale, float(softmax_scale), bool(causal), out)
        return out

    def sage2_prefill_f16_bf16_d128(self, q, k, v, *, softmax_scale=None, causal=False, out=None):
        q_i8, q_scale = self.quantize_q_bf16_d128(q)
        k_i8, k_scale = self.quantize_k_bf16_d128(k)
        v_half = self.quantize_v_fp16_bf16_d128(v)
        return self.sage2_qk_int8_sv_f16_bf16_d128(q_i8, k_i8, v_half, q_scale, k_scale, softmax_scale=softmax_scale, causal=causal, out=out)

    def sage2_prefill_fp8v_bf16_d128(self, q, k, v, *, softmax_scale=None, causal=False, out=None):
        q_i8, q_scale = self.quantize_q_bf16_d128(q)
        k_i8, k_scale = self.quantize_k_bf16_d128(k)
        v_fp8_tpp, v_scale = self.quantize_v_fp8_bf16_d128(v)
        return self.sage2_qk_int8_sv_f8_bf16_d128(q_i8, k_i8, v_fp8_tpp, q_scale, k_scale, v_scale, softmax_scale=softmax_scale, causal=causal, out=out)


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
    suffix = "a" if major >= 12 else ""
    return f"{major}.{minor}{suffix}"


def load_source_ops():
    from torch.utils.cpp_extension import load

    if not REGISTRATION_INCLUDE.is_dir():
        raise RuntimeError(f"missing kernel-builder registration include: {REGISTRATION_INCLUDE}")
    _preload_cublaslt()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _current_arch_list())
    namespace = "sageattention2_blackwell_source_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "sage2_blackwell.cu"),
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
        verbose=False,
    )
    return SourceOps(namespace)


def load_installed_ops(artifact: str | None):
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return importlib.import_module("sageattention2_blackwell")
    finally:
        if artifact:
            sys.path.remove(artifact)


def make_inputs(batch: int, seqlen: int, q_heads: int, kv_heads: int):
    q = (torch.randn(batch, seqlen, q_heads, 128, device="cuda", dtype=torch.float32) * 0.35).to(torch.bfloat16)
    k = (torch.randn(batch, seqlen, kv_heads, 128, device="cuda", dtype=torch.float32) * 0.35).to(torch.bfloat16)
    v = (torch.randn(batch, seqlen, kv_heads, 128, device="cuda", dtype=torch.float32) * 0.35).to(torch.bfloat16)
    return q.contiguous(), k.contiguous(), v.contiguous()


def reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    q_t = q.transpose(1, 2).float()
    if q.shape[2] != k.shape[2]:
        repeat = q.shape[2] // k.shape[2]
        k = k.repeat_interleave(repeat, dim=2)
        v = v.repeat_interleave(repeat, dim=2)
    k_t = k.transpose(1, 2).float()
    v_t = v.transpose(1, 2).float()
    out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=causal)
    return out.transpose(1, 2).to(torch.bfloat16).contiguous()


def stats(got: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    diff = (got.float() - ref.float()).abs().flatten()
    got_f = got.float().flatten()
    ref_f = ref.float().flatten()
    p99_src = diff
    if p99_src.numel() > 8_000_000:
        stride = (p99_src.numel() + 8_000_000 - 1) // 8_000_000
        p99_src = p99_src[::stride]
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "p99_abs": float(torch.quantile(p99_src, 0.99).item()),
        "cos": float(F.cosine_similarity(got_f, ref_f, dim=0).item()),
    }


def run_case(ops, name: str, batch: int, seqlen: int, q_heads: int, kv_heads: int, causal: bool, use_fp8v: bool):
    q, k, v = make_inputs(batch, seqlen, q_heads, kv_heads)
    ref = reference(q, k, v, causal)
    if use_fp8v:
        got = ops.sage2_prefill_fp8v_bf16_d128(q, k, v, causal=causal)
    else:
        got = ops.sage2_prefill_f16_bf16_d128(q, k, v, causal=causal)
    torch.cuda.synchronize()
    s = stats(got, ref)
    min_cos = 0.9985 if seqlen <= 512 else 0.998
    if s["cos"] < min_cos or s["p99_abs"] > 0.25:
        raise AssertionError(f"{name} failed: {s}")
    print(
        f"PASS {name}: max_abs={s['max_abs']:.6f} mean_abs={s['mean_abs']:.6f} "
        f"p99_abs={s['p99_abs']:.6f} cos={s['cos']:.8f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    major, _minor = torch.cuda.get_device_capability(0)
    if major < 12:
        raise SystemExit("sageattention2-blackwell requires Blackwell-class CUDA capability")

    torch.manual_seed(2026)
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    cases = [
        ("wan_noncausal_s128_f16v", 1, 128, 24, 24, False, False),
        ("qwen_causal_gqa_s128_f16v", 1, 128, 32, 8, True, False),
    ]
    if args.mode == "full":
        cases.extend(
            [
                ("wan_noncausal_s256_fp8v", 1, 256, 24, 24, False, True),
                ("qwen_causal_gqa_s256_fp8v", 1, 256, 32, 8, True, True),
                ("qwen_causal_gqa_s512_f16v", 1, 512, 32, 8, True, False),
                ("wan_noncausal_s3600_partial_f16v", 1, 3600, 24, 24, False, False),
                ("wan_noncausal_s3600_partial_fp8v", 1, 3600, 24, 24, False, True),
                ("wan_noncausal_s5070_partial_f16v", 1, 5070, 24, 24, False, False),
                ("qwen_causal_gqa_s3600_partial_f16v", 1, 3600, 32, 8, True, False),
            ]
        )
    for case in cases:
        run_case(ops, *case)


if __name__ == "__main__":
    main()
