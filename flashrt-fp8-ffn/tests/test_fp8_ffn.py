#!/usr/bin/env python3
"""Correctness tests for flashrt-fp8-ffn."""

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
PACKAGE = ROOT / "flashrt-fp8-ffn"
REGISTRATION_INCLUDE = (
    ROOT.parent
    / "kernels"
    / "kernel-builder"
    / "src"
    / "pyproject"
    / "templates"
    / "torch"
)


def fp8_dtype() -> torch.dtype:
    if torch.version.hip is not None and hasattr(torch, "float8_e4m3fnuz"):
        return torch.float8_e4m3fnuz
    return torch.float8_e4m3fn


def fp8_max() -> float:
    return 240.0 if torch.version.hip is not None else 448.0


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self._ops = getattr(torch.ops, namespace)

    def fp8_gemm_bf16(self, x, w, x_scale, w_scale, out=None):
        if out is None:
            out = torch.empty((x.shape[0], w.shape[0]), device=x.device, dtype=torch.bfloat16)
        self._ops.fp8_gemm_bf16(x, w, x_scale, w_scale, out)
        return out

    def fp8_linear_bias_gelu_quant_bf16(
        self, x, w, bias, x_scale, w_scale, y_scale, hidden=None, out=None
    ):
        if hidden is None:
            hidden = torch.empty((x.shape[0], w.shape[0]), device=x.device, dtype=torch.bfloat16)
        if out is None:
            out = torch.empty_like(hidden, dtype=fp8_dtype())
        self._ops.fp8_linear_bias_gelu_quant_bf16(
            x, w, bias, x_scale, w_scale, y_scale, hidden, out
        )
        return hidden, out

    def fp8_gelu_mlp_bf16(
        self,
        x,
        up_w,
        up_b,
        down_w,
        down_b,
        x_scale,
        up_w_scale,
        hidden_scale,
        down_w_scale,
        hidden=None,
        hidden_fp8=None,
        out=None,
    ):
        if hidden is None:
            hidden = torch.empty((x.shape[0], up_w.shape[0]), device=x.device, dtype=torch.bfloat16)
        if hidden_fp8 is None:
            hidden_fp8 = torch.empty_like(hidden, dtype=fp8_dtype())
        if out is None:
            out = torch.empty((x.shape[0], down_w.shape[0]), device=x.device, dtype=torch.bfloat16)
        self._ops.fp8_gelu_mlp_bf16(
            x,
            up_w,
            up_b,
            down_w,
            down_b,
            x_scale,
            up_w_scale,
            hidden_scale,
            down_w_scale,
            hidden,
            hidden_fp8,
            out,
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
    namespace = "flashrt_fp8_ffn_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "fp8_ffn.cu"),
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
        return importlib.import_module("flashrt_fp8_ffn")
    finally:
        if artifact:
            sys.path.remove(artifact)


def quantize_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x.float() / scale.float(), -fp8_max(), fp8_max()).to(fp8_dtype())


def dequant_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x.float() * scale.float()


def ref_gemm(x, w, x_scale, w_scale) -> torch.Tensor:
    return (dequant_fp8(x, x_scale) @ dequant_fp8(w, w_scale).T).to(torch.bfloat16)


def ref_linear_bias_gelu_quant(x, w, bias, x_scale, w_scale, y_scale):
    hidden = ref_gemm(x, w, x_scale, w_scale)
    y = torch.nn.functional.gelu(hidden.float() + bias.float(), approximate="tanh")
    y_fp8 = torch.clamp(y / y_scale.float(), -fp8_max(), fp8_max()).to(fp8_dtype())
    return hidden, y_fp8


def ref_mlp(x, up_w, up_b, down_w, down_b, x_scale, up_w_scale, hidden_scale, down_w_scale):
    _, hidden_fp8 = ref_linear_bias_gelu_quant(
        x, up_w, up_b, x_scale, up_w_scale, hidden_scale
    )
    out = ref_gemm(hidden_fp8, down_w, hidden_scale, down_w_scale)
    return (out.float() + down_b.float()).to(torch.bfloat16)


def make_case(M: int, K: int, H: int, N: int):
    x_scale = torch.tensor([0.05], device="cuda", dtype=torch.float32)
    up_w_scale = torch.tensor([0.04], device="cuda", dtype=torch.float32)
    hidden_scale = torch.tensor([0.25], device="cuda", dtype=torch.float32)
    down_w_scale = torch.tensor([0.04], device="cuda", dtype=torch.float32)
    x = quantize_fp8(torch.randn((M, K), device="cuda", dtype=torch.bfloat16), x_scale)
    up_w = quantize_fp8(torch.randn((H, K), device="cuda", dtype=torch.bfloat16), up_w_scale)
    down_w = quantize_fp8(torch.randn((N, H), device="cuda", dtype=torch.bfloat16), down_w_scale)
    up_b = torch.randn((H,), device="cuda", dtype=torch.bfloat16)
    down_b = torch.randn((N,), device="cuda", dtype=torch.bfloat16)
    return x, up_w, up_b, down_w, down_b, x_scale, up_w_scale, hidden_scale, down_w_scale


def assert_close(name: str, got: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float) -> None:
    max_abs = float((got.float() - expected.float()).abs().max().item())
    if not torch.allclose(got.float(), expected.float(), atol=atol, rtol=rtol):
        raise AssertionError(f"{name} failed: max_abs={max_abs} atol={atol} rtol={rtol}")
    print(f"PASS {name}: max_abs={max_abs:.6f}")


def percentile(x: torch.Tensor, q: float) -> torch.Tensor:
    flat = x.flatten()
    k = max(1, min(flat.numel(), math.ceil(q * flat.numel())))
    return flat.kthvalue(k).values


def assert_distribution_close(
    name: str,
    got: torch.Tensor,
    expected: torch.Tensor,
    *,
    p99_abs_limit: float,
    p99_rel_floor1_limit: float,
) -> None:
    diff = (got.float() - expected.float()).abs().flatten()
    exp = expected.float().abs().flatten().clamp_min(1.0)
    rel = diff / exp
    max_abs = float(diff.max().item())
    p99_abs = float(percentile(diff, 0.99).item())
    max_rel = float(rel.max().item())
    p99_rel = float(percentile(rel, 0.99).item())
    if p99_abs > p99_abs_limit or p99_rel > p99_rel_floor1_limit:
        raise AssertionError(
            f"{name} failed: max_abs={max_abs} p99_abs={p99_abs} "
            f"p99_rel_floor1={p99_rel} max_rel_floor1={max_rel}"
        )
    print(
        f"PASS {name}: max_abs={max_abs:.6f} "
        f"p99_abs={p99_abs:.6f} p99_rel_floor1={p99_rel:.6f} "
        f"max_rel_floor1={max_rel:.6f}"
    )


def assert_fp8_quant_close(
    name: str,
    got: torch.Tensor,
    expected: torch.Tensor,
    *,
    p99_abs_limit: float,
    mismatch_rate_limit: float,
) -> None:
    diff = (got.float() - expected.float()).abs().flatten()
    mismatches = int((got.detach().cpu() != expected.detach().cpu()).sum().item())
    mismatch_rate = mismatches / got.numel()
    max_abs = float(diff.max().item())
    p99_abs = float(percentile(diff, 0.99).item())
    if p99_abs > p99_abs_limit or mismatch_rate > mismatch_rate_limit:
        raise AssertionError(
            f"{name} failed: fp8_max_abs={max_abs} fp8_p99_abs={p99_abs} "
            f"mismatches={mismatches} mismatch_rate={mismatch_rate}"
        )
    print(
        f"PASS {name}: fp8_max_abs={max_abs:.6f} fp8_p99_abs={p99_abs:.6f} "
        f"mismatches={mismatches} mismatch_rate={mismatch_rate:.8f}"
    )


def run(args) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(11)
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)

    for label, shape in {
        "small": (16, 128, 256, 128),
        "pi05_vision": (512, 1152, 4304, 1152),
        "pi05_decoder": (10, 1024, 4096, 1024),
        "groot_vit": (128, 1024, 4096, 1024),
        "groot_deepstack": (128, 4096, 4096, 2048),
        "groot_vl_self_attn_long": (2520, 2048, 8192, 2048),
        "groot_action_dit": (41, 1536, 6144, 1536),
    }.items():
        x, up_w, up_b, down_w, down_b, x_s, up_s, hid_s, dn_s = make_case(*shape)

        got_gemm = ops.fp8_gemm_bf16(x, up_w, x_s, up_s)
        exp_gemm = ref_gemm(x, up_w, x_s, up_s)
        assert_close(f"{label}/fp8_gemm_bf16", got_gemm, exp_gemm, atol=0.25, rtol=0.03)

        _, got_hidden_fp8 = ops.fp8_linear_bias_gelu_quant_bf16(x, up_w, up_b, x_s, up_s, hid_s)
        _, exp_hidden_fp8 = ref_linear_bias_gelu_quant(x, up_w, up_b, x_s, up_s, hid_s)
        assert_fp8_quant_close(
            f"{label}/fp8_linear_bias_gelu_quant_bf16",
            got_hidden_fp8,
            exp_hidden_fp8,
            p99_abs_limit=0.0,
            mismatch_rate_limit=1e-4,
        )

        got_mlp = ops.fp8_gelu_mlp_bf16(x, up_w, up_b, down_w, down_b, x_s, up_s, hid_s, dn_s)
        exp_mlp = ref_mlp(x, up_w, up_b, down_w, down_b, x_s, up_s, hid_s, dn_s)
        assert_distribution_close(
            f"{label}/fp8_gelu_mlp_bf16",
            got_mlp,
            exp_mlp,
            p99_abs_limit=1.0,
            p99_rel_floor1_limit=0.05,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
