#!/usr/bin/env python3
"""Correctness tests for flashrt-residual-norm-quant."""

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
PACKAGE = ROOT / "flashrt-residual-norm-quant"
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

    def rms_norm_bf16(self, x, weight, eps=1e-6, out=None):
        if out is None:
            out = torch.empty_like(x, dtype=torch.bfloat16)
        self._ops.rms_norm_bf16(x, weight, float(eps), out)
        return out

    def rms_norm_quant_fp8_static_bf16(self, x, weight, scale, eps=1e-6, out=None):
        if out is None:
            out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        self._ops.rms_norm_quant_fp8_static_bf16(x, weight, scale, float(eps), out)
        return out

    def layer_norm_bf16(self, x, weight, bias, eps=1e-5, out=None):
        if out is None:
            out = torch.empty_like(x, dtype=torch.bfloat16)
        self._ops.layer_norm_bf16(x, weight, bias, float(eps), out)
        return out

    def residual_add_rms_norm_quant_fp8_static_bf16(
        self, residual, x, weight, scale, eps=1e-6, out=None
    ):
        if out is None:
            out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        self._ops.residual_add_rms_norm_quant_fp8_static_bf16(
            residual, x, weight, scale, float(eps), out
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
    namespace = "flashrt_residual_norm_quant_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "residual_norm_quant.cu"),
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
        return importlib.import_module("flashrt_residual_norm_quant")
    finally:
        if artifact:
            sys.path.remove(artifact)


def quantize_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x.float() / scale.float(), -448.0, 448.0).to(torch.float8_e4m3fn)


def ref_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    rms = torch.rsqrt(torch.mean(x.float() * x.float(), dim=1, keepdim=True) + eps)
    return x.float() * rms * weight.float()


def ref_rms_norm_bf16(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return ref_rms_norm(x, weight, eps).to(torch.bfloat16)


def ref_rms_norm_quant(x, weight, scale, eps) -> torch.Tensor:
    return quantize_fp8(ref_rms_norm(x, weight, eps), scale)


def ref_layer_norm_bf16(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.nn.functional.layer_norm(
        x.float(),
        (x.shape[1],),
        weight.float(),
        bias.float(),
        eps=eps,
    ).to(torch.bfloat16)


def ref_residual_add_rms_norm_quant(residual, x, weight, scale, eps):
    added = residual.float() + x.float()
    residual_out = added.to(torch.bfloat16)
    rms = torch.rsqrt(torch.mean(added * added, dim=1, keepdim=True) + eps)
    # The production kernel rereads the BF16 residual value for the output pass.
    norm = residual_out.float() * rms * weight.float()
    return residual_out, quantize_fp8(norm, scale)


def make_case(rows: int, dim: int):
    x = torch.randn((rows, dim), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn((rows, dim), device="cuda", dtype=torch.bfloat16)
    weight = (1.0 + 0.1 * torch.randn((dim,), device="cuda", dtype=torch.bfloat16)).contiguous()
    bias = (0.1 * torch.randn((dim,), device="cuda", dtype=torch.bfloat16)).contiguous()
    scale = torch.tensor([0.04], device="cuda", dtype=torch.float32)
    return x, residual, weight, bias, scale


def percentile(x: torch.Tensor, q: float) -> torch.Tensor:
    flat = x.flatten()
    k = max(1, min(flat.numel(), math.ceil(q * flat.numel())))
    return flat.kthvalue(k).values


def distribution_metrics(got: torch.Tensor, expected: torch.Tensor):
    diff = (got.float() - expected.float()).abs().flatten()
    rel = diff / expected.float().abs().flatten().clamp_min(1.0)
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "p99_abs": float(percentile(diff, 0.99).item()),
        "p99_rel": float(percentile(rel, 0.99).item()),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                got.float().flatten(), expected.float().flatten(), dim=0
            ).item()
        ),
    }


def assert_close_distribution(
    name: str,
    got: torch.Tensor,
    expected: torch.Tensor,
    *,
    p99_abs_limit: float,
    p99_rel_limit: float,
) -> None:
    m = distribution_metrics(got, expected)
    if m["p99_abs"] > p99_abs_limit or m["p99_rel"] > p99_rel_limit:
        raise AssertionError(
            f"{name} failed: max_abs={m['max_abs']} mean_abs={m['mean_abs']} "
            f"p99_abs={m['p99_abs']} p99_rel={m['p99_rel']} cosine={m['cosine']}"
        )
    print(
        f"PASS {name}: max_abs={m['max_abs']:.6f} mean_abs={m['mean_abs']:.6f} "
        f"p99_abs={m['p99_abs']:.6f} p99_rel={m['p99_rel']:.6f} "
        f"cosine={m['cosine']:.8f}"
    )


def assert_fp8_close(name: str, got: torch.Tensor, expected: torch.Tensor) -> None:
    diff = (got.float() - expected.float()).abs().flatten()
    mismatches = int((got.detach().cpu() != expected.detach().cpu()).sum().item())
    mismatch_rate = mismatches / got.numel()
    max_abs = float(diff.max().item())
    p99_abs = float(percentile(diff, 0.99).item())
    if p99_abs > 0.5 or mismatch_rate > 0.01:
        raise AssertionError(
            f"{name} failed: max_abs={max_abs} p99_abs={p99_abs} "
            f"mismatch_rate={mismatch_rate}"
        )
    print(
        f"PASS {name}: fp8_max_abs={max_abs:.6f} fp8_p99_abs={p99_abs:.6f} "
        f"mismatches={mismatches} mismatch_rate={mismatch_rate:.8f}"
    )


def expect_runtime_error(label: str, fn) -> None:
    try:
        fn()
    except RuntimeError as exc:
        print(f"PASS {label}: rejected invalid input ({str(exc).splitlines()[0]})")
        return
    raise AssertionError(f"{label} failed: expected RuntimeError")


def run_shape(ops, label: str, rows: int, dim: int, eps: float) -> None:
    x, residual, weight, bias, scale = make_case(rows, dim)

    got_norm = ops.rms_norm_bf16(x, weight, eps)
    exp_norm = ref_rms_norm_bf16(x, weight, eps)
    assert_close_distribution(
        f"{label}/rms_norm_bf16",
        got_norm,
        exp_norm,
        p99_abs_limit=0.015625,
        p99_rel_limit=0.02,
    )

    got_ln = ops.layer_norm_bf16(x, weight, bias, eps)
    exp_ln = ref_layer_norm_bf16(x, weight, bias, eps)
    assert_close_distribution(
        f"{label}/layer_norm_bf16",
        got_ln,
        exp_ln,
        p99_abs_limit=0.015625,
        p99_rel_limit=0.02,
    )

    got_fp8 = ops.rms_norm_quant_fp8_static_bf16(x, weight, scale, eps)
    exp_fp8 = ref_rms_norm_quant(x, weight, scale, eps)
    assert_fp8_close(f"{label}/rms_norm_quant_fp8_static_bf16", got_fp8, exp_fp8)

    residual_in = residual.clone()
    got_res_fp8 = ops.residual_add_rms_norm_quant_fp8_static_bf16(
        residual, x, weight, scale, eps
    )
    exp_residual, exp_res_fp8 = ref_residual_add_rms_norm_quant(
        residual_in, x, weight, scale, eps
    )
    assert_close_distribution(
        f"{label}/residual_inplace",
        residual,
        exp_residual,
        p99_abs_limit=0.0,
        p99_rel_limit=0.0,
    )
    assert_fp8_close(
        f"{label}/residual_add_rms_norm_quant_fp8_static_bf16",
        got_res_fp8,
        exp_res_fp8,
    )


def run_rejection_tests(ops) -> None:
    x, residual, weight, bias, scale = make_case(4, 128)
    bad_x = torch.randn((4, 127), device="cuda", dtype=torch.bfloat16)
    bad_weight = torch.randn((127,), device="cuda", dtype=torch.bfloat16)
    bad_out = torch.empty((4, 128), device="cuda", dtype=torch.bfloat16).t()
    bad_residual = torch.randn((4, 64), device="cuda", dtype=torch.bfloat16)

    expect_runtime_error(
        "reject odd hidden dim",
        lambda: ops.rms_norm_bf16(bad_x, bad_weight),
    )
    expect_runtime_error(
        "reject non-contiguous output",
        lambda: ops.rms_norm_bf16(x, weight, out=bad_out),
    )
    expect_runtime_error(
        "reject layer norm bias shape mismatch",
        lambda: ops.layer_norm_bf16(x, weight, bad_weight),
    )
    expect_runtime_error(
        "reject residual shape mismatch",
        lambda: ops.residual_add_rms_norm_quant_fp8_static_bf16(
            bad_residual, x, weight, scale
        ),
    )


def run(args) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(23)
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)

    shapes = {
        "small": (16, 128),
        "pi05_decoder": (10, 1024),
        "pi05_vision": (512, 1152),
        "groot_vl": (1024, 2048),
        "video_prefill": (2520, 2048),
    }
    if args.mode == "smoke":
        shapes = {k: shapes[k] for k in ("small", "pi05_decoder")}

    for label, (rows, dim) in shapes.items():
        run_shape(ops, label, rows, dim, args.eps)
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
