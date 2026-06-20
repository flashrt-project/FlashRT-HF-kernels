#!/usr/bin/env python3
"""Correctness tests for turboquant-kv."""

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
PACKAGE = ROOT / "turboquant-kv"
REGISTRATION_INCLUDE = (
    ROOT.parent / "kernels" / "kernel-builder" / "src" / "pyproject" / "templates" / "torch"
)


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self._ops = getattr(torch.ops, namespace)

    def unpack_packed_bf16(self, k_idx, k_qjl, v_idx, cb_k, cb_v, b_k, b_v):
        y_k = torch.empty((k_idx.shape[0], 256), device=k_idx.device, dtype=torch.bfloat16)
        qjl = torch.empty_like(y_k)
        y_v = torch.empty_like(y_k)
        self._ops.unpack_packed_bf16(k_idx, k_qjl, v_idx, cb_k, cb_v, int(b_k), int(b_v), y_k, qjl, y_v)
        return y_k, qjl, y_v

    def unpack_packed_mixed(self, k_idx, k_qjl, v_idx, cb_k, cb_v, b_k, b_v):
        y_k = torch.empty((k_idx.shape[0], 256), device=k_idx.device, dtype=torch.bfloat16)
        qjl = torch.empty((k_idx.shape[0], 256), device=k_idx.device, dtype=torch.float32)
        y_v = torch.empty_like(y_k)
        self._ops.unpack_packed_mixed(k_idx, k_qjl, v_idx, cb_k, cb_v, int(b_k), int(b_v), y_k, qjl, y_v)
        return y_k, qjl, y_v

    def combine_kv_bf16(self, k_mse, k_qjl, v_unit, k_norm, k_rnorm, v_norm, coef):
        k_out = torch.empty_like(k_mse)
        v_out = torch.empty_like(v_unit)
        self._ops.combine_kv_bf16(k_mse, k_qjl, v_unit, k_norm, k_rnorm, v_norm, float(coef), k_out, v_out)
        return k_out, v_out


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
    namespace = "turboquant_kv_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "turboquant_kv.cu"),
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
        return importlib.import_module("turboquant_kv")
    finally:
        if artifact:
            sys.path.remove(artifact)


def ref_unpack(k_idx, k_qjl, v_idx, cb_k, cb_v, b_k, b_v, qjl_dtype):
    k_lo = k_idx & 0x0F
    k_hi = (k_idx >> 4) & 0x0F
    v_lo = v_idx & 0x0F
    v_hi = (v_idx >> 4) & 0x0F
    k_codes = torch.stack([k_lo, k_hi], dim=-1).reshape(k_idx.shape[0], 256) & ((1 << b_k) - 1)
    v_codes = torch.stack([v_lo, v_hi], dim=-1).reshape(v_idx.shape[0], 256) & ((1 << b_v) - 1)
    bits = torch.arange(8, device=k_qjl.device, dtype=torch.uint8)
    q_bits = ((k_qjl.unsqueeze(-1) >> bits) & 1).reshape(k_qjl.shape[0], 256)
    qjl = torch.where(q_bits.bool(), torch.ones_like(q_bits, dtype=qjl_dtype), -torch.ones_like(q_bits, dtype=qjl_dtype))
    return cb_k[k_codes.long()].to(torch.bfloat16), qjl, cb_v[v_codes.long()].to(torch.bfloat16)


def assert_close(name, got, ref, atol=0.0):
    diff = (got.float() - ref.float()).abs()
    max_err = diff.max().item()
    cos = torch.nn.functional.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0).item()
    if max_err > atol or cos < 0.9999:
        raise AssertionError(f"{name}: max_err={max_err:.8f}, cos={cos:.8f}")


def run_unpack_tests(ops) -> int:
    count = 0
    for m in [1, 4, 128, 1024]:
        for b_k, b_v in [(3, 4), (2, 3), (3, 3), (4, 4)]:
            k_idx = torch.randint(0, 256, (m, 128), device="cuda", dtype=torch.uint8)
            k_qjl = torch.randint(0, 256, (m, 32), device="cuda", dtype=torch.uint8)
            v_idx = torch.randint(0, 256, (m, 128), device="cuda", dtype=torch.uint8)
            cb_k = torch.randn((16,), device="cuda", dtype=torch.float32)
            cb_v = torch.randn((16,), device="cuda", dtype=torch.float32)
            yk, qjl, yv = ops.unpack_packed_bf16(k_idx, k_qjl, v_idx, cb_k, cb_v, b_k, b_v)
            ryk, rqjl, ryv = ref_unpack(k_idx, k_qjl, v_idx, cb_k, cb_v, b_k, b_v, torch.bfloat16)
            assert_close(f"unpack_bf16 yk m={m} bits={b_k}/{b_v}", yk, ryk)
            assert_close(f"unpack_bf16 qjl m={m} bits={b_k}/{b_v}", qjl, rqjl)
            assert_close(f"unpack_bf16 yv m={m} bits={b_k}/{b_v}", yv, ryv)
            yk, qjl_f, yv = ops.unpack_packed_mixed(k_idx, k_qjl, v_idx, cb_k, cb_v, b_k, b_v)
            ryk, rqjl_f, ryv = ref_unpack(k_idx, k_qjl, v_idx, cb_k, cb_v, b_k, b_v, torch.float32)
            assert_close(f"unpack_mixed yk m={m} bits={b_k}/{b_v}", yk, ryk)
            assert_close(f"unpack_mixed qjl m={m} bits={b_k}/{b_v}", qjl_f, rqjl_f)
            assert_close(f"unpack_mixed yv m={m} bits={b_k}/{b_v}", yv, ryv)
            count += 6
    return count


def run_combine_tests(ops) -> int:
    count = 0
    for m in [1, 4, 128, 1024, 4096]:
        k_mse = torch.randn((m, 256), device="cuda", dtype=torch.bfloat16)
        k_qjl = torch.randn((m, 256), device="cuda", dtype=torch.bfloat16)
        v_unit = torch.randn((m, 256), device="cuda", dtype=torch.bfloat16)
        k_norm = torch.rand((m,), device="cuda", dtype=torch.float16) + 0.5
        k_rnorm = torch.rand((m,), device="cuda", dtype=torch.float16) + 0.5
        v_norm = torch.rand((m,), device="cuda", dtype=torch.float16) + 0.5
        coef = 0.125
        k_out, v_out = ops.combine_kv_bf16(k_mse, k_qjl, v_unit, k_norm, k_rnorm, v_norm, coef)
        ref_k = (k_norm.float().unsqueeze(1) * (k_mse.float() + coef * k_rnorm.float().unsqueeze(1) * k_qjl.float())).to(torch.bfloat16)
        ref_v = (v_norm.float().unsqueeze(1) * v_unit.float()).to(torch.bfloat16)
        assert_close(f"combine k m={m}", k_out, ref_k)
        assert_close(f"combine v m={m}", v_out, ref_v)
        count += 2
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
    total = run_unpack_tests(ops) + run_combine_tests(ops)
    torch.cuda.synchronize()
    print(f"turboquant-kv correctness passed: {total} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
