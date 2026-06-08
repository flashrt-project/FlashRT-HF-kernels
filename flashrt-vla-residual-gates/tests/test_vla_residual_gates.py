#!/usr/bin/env python3
"""Correctness tests for flashrt-vla-residual-gates."""

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
PACKAGE = ROOT / "flashrt-vla-residual-gates"
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

    def joint3_bias_gate_residual_bf16(
        self,
        v_residual,
        v_x,
        v_bias,
        v_gate,
        a_residual,
        a_x,
        a_bias,
        a_gate,
        u_residual,
        u_x,
        v_out=None,
        a_out=None,
        u_out=None,
    ):
        if v_out is None:
            v_out = torch.empty_like(v_residual)
        if a_out is None:
            a_out = torch.empty_like(a_residual)
        if u_out is None:
            u_out = torch.empty_like(u_residual)
        self._ops.joint3_bias_gate_residual_bf16(
            v_residual,
            v_x,
            v_bias,
            v_gate,
            v_out,
            a_residual,
            a_x,
            a_bias,
            a_gate,
            a_out,
            u_residual,
            u_x,
            u_out,
        )
        return v_out, a_out, u_out

    def joint3_bias_gate_residual_action_nobias_bf16(
        self,
        v_residual,
        v_x,
        v_bias,
        v_gate,
        a_residual,
        a_x,
        a_gate,
        u_residual,
        u_x,
        v_out=None,
        a_out=None,
        u_out=None,
    ):
        if v_out is None:
            v_out = torch.empty_like(v_residual)
        if a_out is None:
            a_out = torch.empty_like(a_residual)
        if u_out is None:
            u_out = torch.empty_like(u_residual)
        self._ops.joint3_bias_gate_residual_action_nobias_bf16(
            v_residual,
            v_x,
            v_bias,
            v_gate,
            v_out,
            a_residual,
            a_x,
            a_gate,
            a_out,
            u_residual,
            u_x,
            u_out,
        )
        return v_out, a_out, u_out


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
    namespace = "flashrt_vla_residual_gates_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "residual_gates.cu"),
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
        return importlib.import_module("flashrt_vla_residual_gates")
    finally:
        if artifact:
            sys.path.remove(artifact)


def make_segment(rows: int, dim: int):
    residual = torch.randn((rows, dim), device="cuda", dtype=torch.bfloat16)
    x = torch.randn_like(residual)
    gate = torch.randn_like(residual)
    bias = (0.02 * torch.randn((dim,), device="cuda", dtype=torch.bfloat16)).contiguous()
    out = torch.empty_like(residual)
    return residual, x, bias, gate, out


def ref_bias_gate(residual, x, bias, gate):
    return (residual.float() + (x.float() + bias.float().view(1, -1)) * gate.float()).to(torch.bfloat16)


def ref_gate(residual, x, gate):
    return (residual.float() + x.float() * gate.float()).to(torch.bfloat16)


def ref_add(residual, x):
    return (residual.float() + x.float()).to(torch.bfloat16)


def percentile(x: torch.Tensor, q: float) -> torch.Tensor:
    flat = x.flatten()
    k = max(1, min(flat.numel(), math.ceil(q * flat.numel())))
    return flat.kthvalue(k).values


def assert_close_distribution(name: str, got: torch.Tensor, expected: torch.Tensor) -> None:
    diff = (got.float() - expected.float()).abs().flatten()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    p99_abs = float(percentile(diff, 0.99).item())
    cosine = float(torch.nn.functional.cosine_similarity(got.float().flatten(), expected.float().flatten(), dim=0).item())
    if p99_abs > 0.0 or max_abs > 0.0:
        raise AssertionError(
            f"{name} failed: max_abs={max_abs} mean_abs={mean_abs} p99_abs={p99_abs} cosine={cosine}"
        )
    print(
        f"PASS {name}: max_abs={max_abs:.6f} mean_abs={mean_abs:.6f} "
        f"p99_abs={p99_abs:.6f} cosine={cosine:.8f}"
    )


def run_case(ops, label: str, rows: tuple[int, int, int], dim: int) -> None:
    v = make_segment(rows[0], dim)
    a = make_segment(rows[1], dim)
    u_residual = torch.randn((rows[2], dim), device="cuda", dtype=torch.bfloat16)
    u_x = torch.randn_like(u_residual)
    u_out = torch.empty_like(u_residual)
    ops.joint3_bias_gate_residual_bf16(
        v[0], v[1], v[2], v[3],
        a[0], a[1], a[2], a[3],
        u_residual, u_x,
        v_out=v[4], a_out=a[4], u_out=u_out,
    )
    assert_close_distribution(f"{label}/full_v", v[4], ref_bias_gate(v[0], v[1], v[2], v[3]))
    assert_close_distribution(f"{label}/full_a", a[4], ref_bias_gate(a[0], a[1], a[2], a[3]))
    assert_close_distribution(f"{label}/full_u", u_out, ref_add(u_residual, u_x))

    v2 = make_segment(rows[0], dim)
    a2 = make_segment(rows[1], dim)
    u2_residual = torch.randn((rows[2], dim), device="cuda", dtype=torch.bfloat16)
    u2_x = torch.randn_like(u2_residual)
    u2_out = torch.empty_like(u2_residual)
    ops.joint3_bias_gate_residual_action_nobias_bf16(
        v2[0], v2[1], v2[2], v2[3],
        a2[0], a2[1], a2[3],
        u2_residual, u2_x,
        v_out=v2[4], a_out=a2[4], u_out=u2_out,
    )
    assert_close_distribution(f"{label}/nobias_v", v2[4], ref_bias_gate(v2[0], v2[1], v2[2], v2[3]))
    assert_close_distribution(f"{label}/nobias_a", a2[4], ref_gate(a2[0], a2[1], a2[3]))
    assert_close_distribution(f"{label}/nobias_u", u2_out, ref_add(u2_residual, u2_x))


def run(args) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(41)
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    shapes = {
        "small": ((64, 8, 4), 1024),
        "vla_2k": ((2520, 16, 16), 3072),
        "vla_4k": ((4096, 16, 16), 3072),
    }
    if args.mode == "smoke":
        shapes = {"small": shapes["small"]}
    for label, (rows, dim) in shapes.items():
        run_case(ops, label, rows, dim)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
