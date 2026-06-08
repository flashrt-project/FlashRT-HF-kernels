#!/usr/bin/env python3
"""Correctness tests for flashrt-adaptive-norms."""

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
PACKAGE = ROOT / "flashrt-adaptive-norms"
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

    def ada_rms_norm_style_bf16(self, x, weight, style, eps=1e-6, out=None, gate_out=None):
        if out is None:
            out = torch.empty_like(x)
        if gate_out is None:
            gate_out = torch.empty_like(x)
        self._ops.ada_rms_norm_style_bf16(x, weight, style, float(eps), out, gate_out)
        return out, gate_out

    def gate_residual_ada_norm_fp8_static_bf16(
        self, residual, x, gate, weight, style, scale, eps=1e-6, out=None, gate_out=None
    ):
        if out is None:
            out = torch.empty_like(residual, dtype=torch.float8_e4m3fn)
        if gate_out is None:
            gate_out = torch.empty_like(residual)
        self._ops.gate_residual_ada_norm_fp8_static_bf16(
            residual, x, gate, weight, style, scale, float(eps), out, gate_out
        )
        return residual, out, gate_out


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
    namespace = "flashrt_adaptive_norms_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "adaptive_norms.cu"),
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
        return importlib.import_module("flashrt_adaptive_norms")
    finally:
        if artifact:
            sys.path.remove(artifact)


def make_case(rows: int, dim: int):
    x = torch.randn((rows, dim), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    gate = torch.randn_like(x)
    weight = (1.0 + 0.1 * torch.randn((dim,), device="cuda", dtype=torch.bfloat16)).contiguous()
    style = torch.randn((rows, 3 * dim), device="cuda", dtype=torch.bfloat16) * 0.05
    scale = torch.tensor([0.04], device="cuda", dtype=torch.float32)
    return x, residual, gate, weight, style.contiguous(), scale


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float):
    rms = torch.rsqrt(torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + eps)
    return x.float() * rms * weight.float()


def ref_ada(x, weight, style, eps):
    dim = x.shape[1]
    scale = style[:, :dim].float()
    shift = style[:, dim : 2 * dim].float()
    gate = style[:, 2 * dim :].contiguous()
    normed = rms_norm(x, weight, eps)
    return (normed * (1.0 + scale) + shift).to(torch.bfloat16), gate.to(torch.bfloat16)


def ref_gate_residual_fp8(residual, x, gate, weight, style, scale, eps):
    updated = (residual.float() + x.float() * gate.float()).to(torch.bfloat16)
    dim = updated.shape[1]
    style_scale = style[:, :dim].float()
    shift = style[:, dim : 2 * dim].float()
    gate_out = style[:, 2 * dim :].contiguous().to(torch.bfloat16)
    normed = rms_norm(updated, weight, eps)
    fp8 = ((normed * (1.0 + style_scale) + shift) / scale.float().reshape(())).to(torch.float8_e4m3fn)
    return updated, fp8, gate_out


def percentile(x: torch.Tensor, q: float) -> torch.Tensor:
    flat = x.flatten()
    k = max(1, min(flat.numel(), math.ceil(q * flat.numel())))
    return flat.kthvalue(k).values


def assert_close_distribution(name: str, got: torch.Tensor, expected: torch.Tensor, p99_limit: float, max_limit: float) -> None:
    diff = (got.float() - expected.float()).abs().flatten()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    p99_abs = float(percentile(diff, 0.99).item())
    cosine = float(torch.nn.functional.cosine_similarity(got.float().flatten(), expected.float().flatten(), dim=0).item())
    if p99_abs > p99_limit or max_abs > max_limit:
        raise AssertionError(
            f"{name} failed: max_abs={max_abs} mean_abs={mean_abs} p99_abs={p99_abs} cosine={cosine}"
        )
    print(
        f"PASS {name}: max_abs={max_abs:.6f} mean_abs={mean_abs:.6f} "
        f"p99_abs={p99_abs:.6f} cosine={cosine:.8f}"
    )


def assert_fp8_boundary_distribution(name: str, got: torch.Tensor, expected: torch.Tensor) -> None:
    diff = (got.float() - expected.float()).abs().flatten()
    nonzero = int((diff != 0).sum().item())
    max_nonzero = max(8, diff.numel() // 100000)
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    p99_abs = float(percentile(diff, 0.99).item())
    cosine = float(torch.nn.functional.cosine_similarity(got.float().flatten(), expected.float().flatten(), dim=0).item())
    if p99_abs > 0.0 or nonzero > max_nonzero:
        raise AssertionError(
            f"{name} failed: max_abs={max_abs} mean_abs={mean_abs} p99_abs={p99_abs} "
            f"nonzero={nonzero}/{diff.numel()} cosine={cosine}"
        )
    print(
        f"PASS {name}: max_abs={max_abs:.6f} mean_abs={mean_abs:.6f} "
        f"p99_abs={p99_abs:.6f} nonzero={nonzero}/{diff.numel()} cosine={cosine:.8f}"
    )


def run_shape(ops, label: str, rows: int, dim: int, eps: float) -> None:
    x, residual, gate, weight, style, scale = make_case(rows, dim)
    got, got_gate = ops.ada_rms_norm_style_bf16(x, weight, style, eps)
    exp, exp_gate = ref_ada(x, weight, style, eps)
    assert_close_distribution(f"{label}/ada_out", got, exp, 0.0, 0.015625)
    assert_close_distribution(f"{label}/ada_gate", got_gate, exp_gate, 0.0, 0.0)

    residual_input = residual.clone()
    got_residual, got_fp8, got_gate2 = ops.gate_residual_ada_norm_fp8_static_bf16(
        residual, x, gate, weight, style, scale, eps
    )
    exp_residual, exp_fp8, exp_gate2 = ref_gate_residual_fp8(
        residual_input, x, gate, weight, style, scale, eps
    )
    assert_close_distribution(f"{label}/fused_residual", got_residual, exp_residual, 0.0, 0.0)
    assert_fp8_boundary_distribution(f"{label}/fused_fp8", got_fp8.float(), exp_fp8.float())
    assert_close_distribution(f"{label}/fused_gate", got_gate2, exp_gate2, 0.0, 0.0)


def run(args) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(47)
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    shapes = {
        "small": (64, 1024),
        "vla_2k": (2520, 3072),
        "vla_4k": (4096, 3072),
    }
    if args.mode == "smoke":
        shapes = {"small": shapes["small"]}
    for label, (rows, dim) in shapes.items():
        run_shape(ops, label, rows, dim, args.eps)


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
