#!/usr/bin/env python3
"""Correctness tests for blockwise-fp8-producers."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "blockwise-fp8-producers"
REGISTRATION_INCLUDE = (
    ROOT.parent / "kernels" / "kernel-builder" / "src" / "pyproject" / "templates" / "torch"
)


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self.ops = getattr(torch.ops, namespace)

    @staticmethod
    def _allocate(x):
        return (
            torch.empty_like(x, dtype=torch.float8_e4m3fn),
            torch.empty((x.shape[0], x.shape[1] // 128), device=x.device, dtype=torch.float32),
        )

    def quantize_fp8_block128_bf16(self, x, output=None, scale=None):
        output, scale = self._allocate(x) if output is None else (output, scale)
        self.ops.quantize_fp8_block128_bf16(x, output, scale)
        return output, scale

    def layer_norm_fp8_block128_bf16(self, x, weight, bias, eps=1e-6, output=None, scale=None):
        output, scale = self._allocate(x) if output is None else (output, scale)
        self.ops.layer_norm_fp8_block128_bf16(x, weight, bias, float(eps), output, scale)
        return output, scale

    def rms_norm_fp8_block128_bf16(self, x, weight, eps=1e-6, output=None, scale=None):
        output, scale = self._allocate(x) if output is None else (output, scale)
        self.ops.rms_norm_fp8_block128_bf16(x, weight, float(eps), output, scale)
        return output, scale

    def residual_add_rms_norm_fp8_block128_bf16(
        self, residual, x, weight, eps=1e-6, residual_out=None, output=None, scale=None
    ):
        residual_out = torch.empty_like(x) if residual_out is None else residual_out
        output, scale = self._allocate(x) if output is None else (output, scale)
        self.ops.residual_add_rms_norm_fp8_block128_bf16(
            residual, x, weight, float(eps), residual_out, output, scale
        )
        return residual_out, output, scale

    def gelu_tanh_fp8_block128_bf16(self, x, output=None, scale=None):
        output, scale = self._allocate(x) if output is None else (output, scale)
        self.ops.gelu_tanh_fp8_block128_bf16(x, output, scale)
        return output, scale

    def gelu_tanh_bias_fp8_block128_bf16(self, x, bias, output=None, scale=None):
        output, scale = self._allocate(x) if output is None else (output, scale)
        self.ops.gelu_tanh_bias_fp8_block128_bf16(x, bias, output, scale)
        return output, scale

    def silu_mul_fp8_block128_bf16(self, gate, up, output=None, scale=None):
        output, scale = self._allocate(gate) if output is None else (output, scale)
        self.ops.silu_mul_fp8_block128_bf16(gate, up, output, scale)
        return output, scale

    def silu_mul_merged_fp8_block128_bf16(self, gate_up, output=None, scale=None):
        rows, merged = gate_up.shape
        dim = merged // 2
        if output is None:
            output = torch.empty((rows, dim), device=gate_up.device, dtype=torch.float8_e4m3fn)
            scale = torch.empty((rows, dim // 128), device=gate_up.device, dtype=torch.float32)
        self.ops.silu_mul_merged_fp8_block128_bf16(gate_up, output, scale)
        return output, scale


def _arch_list() -> str:
    major, minor = torch.cuda.get_device_capability(0)
    return "12.0a" if major >= 12 else f"{major}.{minor}"


def load_source_ops() -> SourceOps:
    from torch.utils.cpp_extension import load

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _arch_list())
    namespace = "blockwise_fp8_producers_source_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "fp8_per_token_block_quant.cu"),
            str(PACKAGE / "csrc" / "norm_act_to_fp8_block128.cu"),
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
        is_python_module=False,
        verbose=False,
    )
    return SourceOps(namespace)


def load_installed_ops(artifact: str | None):
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return importlib.import_module("blockwise_fp8_producers")
    finally:
        if artifact:
            sys.path.remove(artifact)


def quantize_ref(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, dim = x.shape
    blocks = x.float().reshape(rows, dim // 128, 128)
    scale = torch.clamp(blocks.abs().amax(dim=-1) / 448.0, min=1.0e-12)
    quant = torch.clamp(blocks / scale.unsqueeze(-1), -448.0, 448.0)
    return quant.to(torch.float8_e4m3fn).reshape_as(x), scale


def dequantize(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (
        x.float().reshape(x.shape[0], x.shape[1] // 128, 128)
        * scale.unsqueeze(-1)
    ).reshape_as(x)


def metrics(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float, float]:
    diff = (got.float() - ref.float()).abs()
    p99 = torch.quantile(diff.flatten(), 0.99).item()
    cos = F.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0).item()
    return diff.max().item(), p99, diff.mean().item(), cos


def assert_quantized(
    name: str, got, scale, producer_ref, *, exact_codes: bool = False
) -> None:
    ref_q, ref_scale = quantize_ref(producer_ref)
    code_mismatch = (got.float() != ref_q.float()).float().mean().item()
    scale_rel = (
        (scale - ref_scale).abs() / ref_scale.abs().clamp_min(1.0e-12)
    ).max().item()
    if exact_codes:
        torch.testing.assert_close(scale, ref_scale, rtol=2e-6, atol=1e-8)
        if code_mismatch > 0.005:
            raise AssertionError(
                f"{name}: FP8 bin mismatch fraction {code_mismatch:.8f} exceeds 0.5%"
            )
    got_deq = dequantize(got, scale)
    max_abs, p99, mean_abs, cosine = metrics(got_deq, producer_ref.float())
    ref_peak = producer_ref.float().abs().max().item()
    ref_mean = producer_ref.float().abs().mean().item()
    max_rel_peak = max_abs / max(ref_peak, 1.0e-12)
    p99_rel_peak = p99 / max(ref_peak, 1.0e-12)
    mean_rel = mean_abs / max(ref_mean, 1.0e-12)
    print(
        f"{name}: max={max_abs:.8f} p99={p99:.8f} "
        f"mean={mean_abs:.8f} cosine={cosine:.8f} scale_rel={scale_rel:.8f} "
        f"code_mismatch={code_mismatch:.8f} "
        f"max/peak={max_rel_peak:.8f} p99/peak={p99_rel_peak:.8f} "
        f"mean/mean={mean_rel:.8f}"
    )
    if (
        scale_rel > 0.01
        or max_rel_peak > 0.04
        or p99_rel_peak > 0.03
        or mean_rel > 0.03
        or cosine < 0.999
    ):
        raise AssertionError(f"{name}: packaged quantization differs from reference")


def expect_runtime_error(name: str, fn) -> None:
    try:
        fn()
    except RuntimeError:
        print(f"{name}: rejected")
        return
    raise AssertionError(f"{name}: expected RuntimeError")


def run(ops, mode: str) -> int:
    torch.manual_seed(47)
    shapes = [(1, 1024), (17, 1152)] if mode == "smoke" else [
        (1, 1024),
        (2, 1280),
        (17, 1152),
        (40, 1536),
        (49, 2048),
        (51, 4096),
        (65, 4352),
        (105, 8192),
        (277, 9216),
        (512, 12288),
    ]
    count = 0
    for rows, dim in shapes:
        x = (torch.randn((rows, dim), device="cuda", dtype=torch.bfloat16) * 0.5).contiguous()
        weight = (torch.randn((dim,), device="cuda", dtype=torch.bfloat16) * 0.2 + 1).contiguous()
        bias = (torch.randn((dim,), device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()

        got, scale = ops.quantize_fp8_block128_bf16(x)
        assert_quantized(
            f"quant rows={rows} dim={dim}", got, scale, x, exact_codes=True
        )
        count += 1

        got, scale = ops.layer_norm_fp8_block128_bf16(x, weight, bias)
        layer_ref = F.layer_norm(x.float(), (dim,), weight.float(), bias.float(), 1e-6).to(torch.bfloat16)
        assert_quantized(f"layer_norm rows={rows} dim={dim}", got, scale, layer_ref)
        count += 1

        got, scale = ops.rms_norm_fp8_block128_bf16(x, weight)
        rms = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + 1e-6)
        rms_ref = (x.float() * rms * weight.float()).to(torch.bfloat16)
        assert_quantized(f"rms_norm rows={rows} dim={dim}", got, scale, rms_ref)
        count += 1

        residual = (torch.randn_like(x) * 0.5).contiguous()
        residual_out, got, scale = ops.residual_add_rms_norm_fp8_block128_bf16(
            residual, x, weight
        )
        residual_ref = (residual.float() + x.float()).to(torch.bfloat16)
        torch.testing.assert_close(residual_out, residual_ref, rtol=0, atol=0)
        rms = torch.rsqrt(residual_ref.float().square().mean(dim=-1, keepdim=True) + 1e-6)
        residual_norm_ref = (residual_ref.float() * rms * weight.float()).to(torch.bfloat16)
        assert_quantized(
            f"residual_rms_norm rows={rows} dim={dim}", got, scale, residual_norm_ref
        )
        count += 2

        got, scale = ops.gelu_tanh_fp8_block128_bf16(x)
        gelu_ref = F.gelu(x.float(), approximate="tanh")
        assert_quantized(f"gelu rows={rows} dim={dim}", got, scale, gelu_ref)
        count += 1

        got, scale = ops.gelu_tanh_bias_fp8_block128_bf16(x, bias)
        gelu_bias_ref = F.gelu(x.float() + bias.float(), approximate="tanh")
        assert_quantized(
            f"gelu_bias rows={rows} dim={dim}", got, scale, gelu_bias_ref
        )
        count += 1

        up = (torch.randn_like(x) * 0.5).contiguous()
        got, scale = ops.silu_mul_fp8_block128_bf16(x, up)
        silu_ref = (F.silu(x.float()).to(torch.bfloat16).float() * up.float()).to(torch.bfloat16)
        assert_quantized(f"silu rows={rows} dim={dim}", got, scale, silu_ref)
        count += 1

        merged = torch.cat((x, up), dim=-1).contiguous()
        merged_got, merged_scale = ops.silu_mul_merged_fp8_block128_bf16(merged)
        torch.testing.assert_close(merged_got.float(), got.float(), rtol=0, atol=0)
        torch.testing.assert_close(merged_scale, scale, rtol=0, atol=0)
        count += 2

    bad = torch.randn((4, 4304), device="cuda", dtype=torch.bfloat16)
    expect_runtime_error("raw 4304 width", lambda: ops.quantize_fp8_block128_bf16(bad))
    expect_runtime_error(
        "noncontiguous input",
        lambda: ops.quantize_fp8_block128_bf16(
            torch.randn((1024, 4), device="cuda", dtype=torch.bfloat16).transpose(0, 1)
        ),
    )
    count += 2
    return count


def run_compile(ops) -> int:
    x = torch.randn((17, 1152), device="cuda", dtype=torch.bfloat16)
    weight = torch.ones((1152,), device="cuda", dtype=torch.bfloat16)
    bias = torch.zeros_like(weight)

    def invoke(a, w, b):
        return ops.layer_norm_fp8_block128_bf16(a, w, b)

    compiled = torch.compile(invoke, fullgraph=True)
    eager_out, eager_scale = invoke(x, weight, bias)
    got_out, got_scale = compiled(x, weight, bias)
    torch.testing.assert_close(got_out.float(), eager_out.float(), rtol=0, atol=0)
    torch.testing.assert_close(got_scale, eager_scale, rtol=0, atol=0)
    print("compile fullgraph: exact")
    return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    count = run(ops, args.mode)
    if args.backend == "installed":
        count += run_compile(ops)
    print(f"blockwise-fp8-producers {args.backend} {args.mode}: passed {count}/{count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
