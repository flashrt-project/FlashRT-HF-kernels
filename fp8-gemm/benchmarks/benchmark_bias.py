#!/usr/bin/env python3
"""Benchmark SM110 BF16-output FP8 GEMM epilogues."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import statistics
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "fp8-gemm"
REGISTRATION_INCLUDE = (
    ROOT.parent
    / "kernels"
    / "kernel-builder"
    / "src"
    / "pyproject"
    / "templates"
    / "torch"
)

SHAPES = {
    "siglip_qkv": (512, 1152, 3456),
    "siglip_mlp_up": (768, 1152, 4304),
    "siglip_mlp_down": (768, 4304, 1152),
}


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self.ops = getattr(torch.ops, namespace)

    def fp8_linear_bias_bf16(self, x, w, bias, alpha=1.0, out=None):
        if out is None:
            out = torch.empty(
                (x.shape[0], w.shape[0]), device=x.device, dtype=torch.bfloat16
            )
        self.ops.fp8_linear_bias_bf16(x, w, bias, float(alpha), out)
        return out

    def fp8_linear_bias_residual_bf16(
        self, x, w, bias, residual, alpha=1.0
    ):
        self.ops.fp8_linear_bias_residual_bf16(
            x, w, bias, float(alpha), residual
        )
        return residual

    def fp8_linear_bias_gelu_bf16(self, x, w, bias, alpha=1.0, out=None):
        if out is None:
            out = torch.empty(
                (x.shape[0], w.shape[0]), device=x.device, dtype=torch.bfloat16
            )
        self.ops.fp8_linear_bias_gelu_bf16(x, w, bias, float(alpha), out)
        return out


def load_source_ops() -> SourceOps:
    from torch.utils.cpp_extension import load

    cutlass = Path(os.environ["CUTLASS_INCLUDE"])
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "11.0a")
    namespace = "fp8_gemm_bias_source_bench"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "cutlass_sm110_fp8_gemm.cu"),
            str(PACKAGE / "csrc" / "cublaslt_fp8_bias_sm110.cu"),
        ],
        extra_include_paths=[
            str(PACKAGE / "csrc"),
            str(REGISTRATION_INCLUDE),
            str(cutlass),
            str(cutlass.parent / "tools" / "util" / "include"),
        ],
        extra_cflags=[
            "-O3", "-DNDEBUG", "-DCUDA_KERNEL",
            "-DFLASHRT_FP8_GEMM_SOURCE_SM110_ONLY",
        ],
        extra_cuda_cflags=[
            "-O3", "-DNDEBUG", "--expt-relaxed-constexpr", "--use_fast_math",
            "-DCUDA_KERNEL", "-DFLASHRT_FP8_GEMM_SOURCE_SM110_ONLY",
        ],
        verbose=False,
    )
    return SourceOps(namespace)


def load_installed_ops(artifact: str | None):
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return importlib.import_module("fp8_gemm")
    finally:
        if artifact:
            sys.path.remove(artifact)


def load_native():
    root = os.environ.get("FLASHRT_NATIVE_ROOT")
    if not root:
        return None
    sys.path.insert(0, root)
    try:
        module = importlib.import_module("flash_rt.flash_rt_kernels")
        return module.GemmRunner()
    finally:
        sys.path.remove(root)


def measure(fn, warmup: int, iterations: int, rounds: int = 7) -> float:
    samples = []
    for _ in range(rounds):
        for _ in range(warmup):
            fn()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iterations)
    return float(statistics.median(samples))


def metrics(got, expected):
    diff = (got.float() - expected.float()).abs().flatten()
    rank = max(1, math.ceil(0.99 * diff.numel()))
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "p99_abs": float(diff.kthvalue(rank).values.item()),
        "cosine": float(torch.nn.functional.cosine_similarity(
            got.float().flatten(), expected.float().flatten(), dim=0
        ).item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (11, 0):
        raise SystemExit("SM110 is required")
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    native = load_native()
    native_stream = int(torch.cuda.current_stream().cuda_stream)
    rows = []
    for name, (m, k, n) in SHAPES.items():
        generator = torch.Generator(device="cuda").manual_seed(m + k + n)
        x = (torch.randn((m, k), device="cuda", generator=generator) * 0.25).to(
            torch.float8_e4m3fn
        )
        weight = (
            torch.randn((n, k), device="cuda", generator=generator) * 0.25
        ).to(torch.float8_e4m3fn)
        weight_kn = weight.t().contiguous()
        bias = (torch.randn((n,), device="cuda", generator=generator) * 0.1).to(
            torch.bfloat16
        )
        alpha = 0.75
        base = x.float() @ weight.float().t() * alpha
        for epilogue in ("bias", "bias_residual", "bias_gelu"):
            if epilogue == "bias":
                out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
                invoke = lambda: ops.fp8_linear_bias_bf16(
                    x, weight, bias, alpha=alpha, out=out
                )
                expected = (base + bias.float()).to(torch.bfloat16)
            elif epilogue == "bias_residual":
                initial = (torch.randn((m, n), device="cuda", generator=generator) * 0.1).to(
                    torch.bfloat16
                )
                out = initial.clone()
                invoke = lambda: ops.fp8_linear_bias_residual_bf16(
                    x, weight, bias, out, alpha=alpha
                )
                expected = (initial.float() + base + bias.float()).to(torch.bfloat16)
            else:
                out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
                invoke = lambda: ops.fp8_linear_bias_gelu_bf16(
                    x, weight, bias, alpha=alpha, out=out
                )
                expected = torch.nn.functional.gelu(
                    base + bias.float(), approximate="tanh"
                ).to(torch.bfloat16)
            invoke()
            torch.cuda.synchronize()
            accuracy = metrics(out, expected)
            hub_us = measure(invoke, args.warmup, args.iterations)
            row = {
                "shape": name,
                "M": m,
                "K": k,
                "N": n,
                "epilogue": epilogue,
                "hub_us": hub_us,
                **accuracy,
            }
            if native is not None and epilogue == "bias":
                native_out = torch.empty_like(out)
                native_invoke = lambda: native.fp8_nn_bias_bf16(
                    x.data_ptr(), weight_kn.data_ptr(), native_out.data_ptr(),
                    bias.data_ptr(), m, n, k, alpha, native_stream
                )
                native_invoke()
                torch.cuda.synchronize()
                row["native_us"] = measure(
                    native_invoke, args.warmup, args.iterations
                )
                row["hub_over_native"] = row["hub_us"] / row["native_us"]
                row["native_metrics"] = metrics(native_out, expected)
            rows.append(row)
    payload = {"device": torch.cuda.get_device_name(), "rows": rows}
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n")
    failed = [
        row for row in rows
        if row["p99_abs"] > 0.25 or row["mean_abs"] > 0.02 or row["cosine"] < 0.999
    ]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
