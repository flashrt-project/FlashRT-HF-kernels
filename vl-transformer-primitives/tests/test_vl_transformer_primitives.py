#!/usr/bin/env python3
"""Correctness tests for vl-transformer-primitives."""

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
PACKAGE = ROOT / "vl-transformer-primitives"
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

    def qwen3_q_norm_rope_qstage_bf16(self, q_pre, q_norm_weight, cos, sin, eps=1e-6):
        out = torch.empty_like(q_pre)
        self._ops.qwen3_q_norm_rope_qstage_bf16(
            q_pre, q_norm_weight, cos, sin, float(eps), out
        )
        return out

    def qwen3_k_norm_rope_kvwrite_bf16(self, k_pre, v_pre, k_norm_weight, cos, sin, eps=1e-6):
        k_out = torch.empty_like(k_pre)
        v_out = torch.empty_like(v_pre)
        self._ops.qwen3_k_norm_rope_kvwrite_bf16(
            k_pre, v_pre, k_norm_weight, cos, sin, float(eps), k_out, v_out
        )
        return k_out, v_out

    def qwen3_k_norm_rope_kvwrite_devpos_bf16(
        self, k_pre, v_pre, k_norm_weight, cos, sin, cur_pos, k_cache, v_cache, eps=1e-6
    ):
        self._ops.qwen3_k_norm_rope_kvwrite_devpos_bf16(
            k_pre, v_pre, k_norm_weight, cos, sin, cur_pos, float(eps), k_cache, v_cache
        )
        return k_cache, v_cache

    def avg_pool_vision_tokens_bf16(self, x, nv, h, w, pool_factor):
        out = torch.empty(
            (nv * (h // pool_factor) * (w // pool_factor), x.shape[1]),
            device=x.device,
            dtype=x.dtype,
        )
        self._ops.avg_pool_vision_tokens_bf16(
            x, int(nv), int(h), int(w), int(pool_factor), out
        )
        return out


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
    namespace = "vl_transformer_primitives_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "qwen3_qkv_post_proc.cu"),
            str(PACKAGE / "csrc" / "avg_pool_vision_tokens.cu"),
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
        return importlib.import_module("vl_transformer_primitives")
    finally:
        if artifact:
            sys.path.remove(artifact)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().flatten()
    bf = b.float().flatten()
    return torch.nn.functional.cosine_similarity(af, bf, dim=0).item()


def assert_close(name: str, got: torch.Tensor, ref: torch.Tensor, atol: float = 0.015625) -> None:
    diff = (got.float() - ref.float()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    cos = cosine(got, ref)
    if max_err > atol or cos < 0.9999:
        raise AssertionError(
            f"{name}: max_err={max_err:.8f}, mean_err={mean_err:.8f}, cos={cos:.8f}"
        )


def ref_norm_rope(x: torch.Tensor, weight: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, eps: float):
    xf = x.float()
    wf = weight.float()
    rstd = torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + eps)
    y = xf * rstd * wf
    lo, hi = y[:, :64], y[:, 64:]
    cf = cos.float().unsqueeze(0)
    sf = sin.float().unsqueeze(0)
    return torch.cat([lo * cf - hi * sf, hi * cf + lo * sf], dim=-1).to(torch.bfloat16)


def ref_avg_pool(x: torch.Tensor, nv: int, h: int, w: int, pool: int):
    dim = x.shape[1]
    return (
        x.float()
        .reshape(nv, h, w, dim)
        .reshape(nv, h // pool, pool, w // pool, pool, dim)
        .mean(dim=(2, 4))
        .reshape(nv * (h // pool) * (w // pool), dim)
        .to(torch.bfloat16)
    )


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


def run_decode_tests(ops) -> int:
    count = 0
    for heads in [1, 4, 8, 16, 32, 40]:
        q, k, v, q_w, k_w, cos, sin = make_decode_case(heads)
        q_got = ops.qwen3_q_norm_rope_qstage_bf16(q, q_w, cos, sin)
        q_ref = ref_norm_rope(q, q_w, cos, sin, 1e-6)
        assert_close(f"q_norm_rope heads={heads}", q_got, q_ref)

        k_got, v_got = ops.qwen3_k_norm_rope_kvwrite_bf16(k, v, k_w, cos, sin)
        k_ref = ref_norm_rope(k, k_w, cos, sin, 1e-6)
        assert_close(f"k_norm_rope heads={heads}", k_got, k_ref)
        assert_close(f"v_copy heads={heads}", v_got, v, atol=0.0)

        k_cache = torch.zeros((8, heads, 128), device="cuda", dtype=torch.bfloat16)
        v_cache = torch.zeros_like(k_cache)
        cur_pos = torch.tensor([3], device="cuda", dtype=torch.int32)
        ops.qwen3_k_norm_rope_kvwrite_devpos_bf16(k, v, k_w, cos, sin, cur_pos, k_cache, v_cache)
        assert_close(f"k_devpos heads={heads}", k_cache[3], k_ref)
        assert_close(f"v_devpos heads={heads}", v_cache[3], v, atol=0.0)
        count += 4
    return count


def run_pool_tests(ops) -> int:
    count = 0
    for nv, h, w, dim, pool in [
        (1, 16, 16, 1024, 2),
        (2, 16, 16, 1152, 2),
        (4, 16, 16, 2048, 4),
        (2, 32, 32, 1024, 4),
    ]:
        x = torch.randn((nv * h * w, dim), device="cuda", dtype=torch.bfloat16)
        got = ops.avg_pool_vision_tokens_bf16(x, nv, h, w, pool)
        ref = ref_avg_pool(x, nv, h, w, pool)
        assert_close(f"avg_pool nv={nv} h={h} w={w} dim={dim} pool={pool}", got, ref)
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
    total = run_decode_tests(ops) + run_pool_tests(ops)
    torch.cuda.synchronize()
    print(f"vl-transformer-primitives correctness passed: {total} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
