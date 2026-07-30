#!/usr/bin/env python3
"""World-model Conv benchmark with native, wrapper, compile, and cuDNN paths."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "tests"))
from test_world_model_conv import (  # noqa: E402
    dequantize_linear_nvfp4,
    load_installed_ops,
    load_source_ops,
    quantize_conv_tensor,
)


CONV3D_SHAPES = {
    "causal-c32": (1, 2, 4, 16, 16, 32, 32),
    "causal-small": (1, 2, 4, 16, 16, 64, 64),
}
NVFP4_CONV3D_SHAPES = {
    "nvfp4-c64": (1, 2, 4, 16, 16, 64, 64),
    "nvfp4-c128": (1, 2, 4, 16, 16, 128, 128),
    "nvfp4-c512": (1, 2, 4, 16, 16, 512, 512),
}
CONV2D_SHAPES = {
    "resample-c64": (4, 32, 32, 64, 64),
    "resample-c320": (17, 32, 32, 320, 320),
}


@dataclass
class Result:
    workload: str
    shape: str
    native_us: float
    wrapper_us: float
    wrapper_native: float
    eager_cudnn_us: float
    compile_cudnn_us: float
    diagnostic_predequant_cudnn_us: float | None
    diagnostic_predequant_compile_us: float | None
    max_abs: float
    mean_abs: float
    p99_abs: float
    cosine: float
    accepted: bool


def bench(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def build_native():
    from torch.utils.cpp_extension import load

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
    return load(
        name="world_model_conv_raw_native",
        sources=[
            str(PACKAGE / "benchmarks/native_binding.cpp"),
            str(PACKAGE / "csrc/fp8_conv3d_sm120_v18.cu"),
            str(PACKAGE / "csrc/fp8_causal_conv3d_sm120.cu"),
            str(PACKAGE / "csrc/fp8_conv2d_3x3_sm120.cu"),
            str(PACKAGE / "csrc/nvfp4_causal_conv3d_sm120.cu"),
            str(PACKAGE / "csrc/nvfp4_causal_conv3d_residual_sm120.cu"),
            str(PACKAGE / "csrc/nvfp4_causal_conv3d_residual_k128_sm120.cu"),
        ],
        extra_include_paths=[str(PACKAGE / "csrc")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


def metrics(got, ref):
    diff = (got.float() - ref.float()).abs().flatten()
    cosine = F.cosine_similarity(
        got.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    return (
        diff.max().item(),
        diff.mean().item(),
        torch.quantile(diff, 0.99).item(),
        cosine,
    )


def source_call(ops, name, *args, out):
    if hasattr(ops, "_ops"):
        getattr(ops._ops, name)(*args, out)
    else:
        getattr(ops, name)(*args, out=out)


def run_conv3d(ops, native, label, shape, args):
    n, tc, tn, h, w, ci, co = shape
    cache = (torch.randn((n, tc, h, w, ci), device="cuda") * 0.1).to(
        torch.float8_e4m3fn
    )
    new = (torch.randn((n, tn, h, w, ci), device="cuda") * 0.1).to(
        torch.float8_e4m3fn
    )
    weight = (torch.randn((co, 3, 3, 3, ci), device="cuda") * 0.1).to(
        torch.float8_e4m3fn
    )
    bias = (torch.randn(co, device="cuda") * 0.01).to(torch.bfloat16)
    out = torch.empty((n, tn, h, w, co), device="cuda", dtype=torch.bfloat16)
    alpha = 0.75

    wrapper = lambda: source_call(
        ops,
        "fp8_causal_conv3d_ndhwc_bf16",
        cache,
        new,
        weight,
        bias,
        alpha,
        out=out,
    )
    raw = lambda: native.causal_conv3d(
        cache, new, weight, bias, alpha, out
    )

    def cudnn_ref():
        x = torch.cat((cache, new), dim=1).float().permute(0, 4, 1, 2, 3)
        wt = weight.float().permute(0, 4, 1, 2, 3)
        y = F.conv3d(x, wt, padding=(0, 1, 1))
        y = y.mul(alpha).add(bias.float().view(1, -1, 1, 1, 1))
        out.copy_(y[:, :, :tn].permute(0, 2, 3, 4, 1).to(torch.bfloat16))

    compiled = torch.compile(cudnn_ref, fullgraph=True)
    wrapper()
    got = out.clone()
    cudnn_ref()
    ref = out.clone()
    max_abs, mean_abs, p99_abs, cosine = metrics(got, ref)
    native_us = bench(raw, args.warmup, args.iters)
    wrapper_us = bench(wrapper, args.warmup, args.iters)
    eager_us = bench(cudnn_ref, args.warmup, args.iters)
    compile_us = bench(compiled, args.warmup, args.iters)
    return Result(
        label,
        str(shape),
        native_us,
        wrapper_us,
        wrapper_us / native_us,
        eager_us,
        compile_us,
        None,
        None,
        max_abs,
        mean_abs,
        p99_abs,
        cosine,
        wrapper_us - native_us <= max(0.5, native_us * 0.05)
        and wrapper_us <= min(eager_us, compile_us) * 0.98
        and cosine >= 0.999
        and mean_abs <= 0.01,
    )


def run_conv3d_residual(ops, native, label, shape, args):
    n, tc, tn, h, w, ci, co = shape
    cache = (torch.randn((n, tc, h, w, ci), device="cuda") * 0.1).to(
        torch.float8_e4m3fn
    )
    new = (torch.randn((n, tn, h, w, ci), device="cuda") * 0.1).to(
        torch.float8_e4m3fn
    )
    weight = (torch.randn((co, 3, 3, 3, ci), device="cuda") * 0.1).to(
        torch.float8_e4m3fn
    )
    bias = (torch.randn(co, device="cuda") * 0.01).to(torch.bfloat16)
    residual = torch.randn(
        (n, co, tn, h, w), device="cuda", dtype=torch.bfloat16
    )
    out = torch.empty_like(residual)
    alpha = 0.75
    wrapper = lambda: source_call(
        ops,
        "fp8_conv3d_v18_ncdhw_res_bf16out",
        cache,
        new,
        weight,
        bias,
        residual,
        alpha,
        out=out,
    )
    raw = lambda: native.causal_conv3d_residual(
        cache, new, weight, bias, residual, alpha, out
    )

    def cudnn_ref():
        x = torch.cat((cache, new), dim=1).float().permute(0, 4, 1, 2, 3)
        wt = weight.float().permute(0, 4, 1, 2, 3)
        y = F.conv3d(x, wt, padding=(0, 1, 1))
        y = y.mul(alpha).add(bias.float().view(1, -1, 1, 1, 1))
        y = (
            y[:, :, :tn].to(torch.bfloat16).float()
            + residual.float()
        ).to(torch.bfloat16)
        out.copy_(y)

    compiled = torch.compile(cudnn_ref, fullgraph=True)
    wrapper()
    got = out.clone()
    cudnn_ref()
    ref = out.clone()
    max_abs, mean_abs, p99_abs, cosine = metrics(got, ref)
    native_us = bench(raw, args.warmup, args.iters)
    wrapper_us = bench(wrapper, args.warmup, args.iters)
    eager_us = bench(cudnn_ref, args.warmup, args.iters)
    compile_us = bench(compiled, args.warmup, args.iters)
    return Result(
        f"{label}-residual",
        str(shape),
        native_us,
        wrapper_us,
        wrapper_us / native_us,
        eager_us,
        compile_us,
        None,
        None,
        max_abs,
        mean_abs,
        p99_abs,
        cosine,
        wrapper_us - native_us <= max(0.5, native_us * 0.05)
        and wrapper_us <= min(eager_us, compile_us) * 0.98
        and cosine >= 0.999
        and mean_abs <= 0.01,
    )


def run_conv2d(ops, native, label, shape, args):
    n, h, w, ci, co = shape
    input = (torch.randn((n, h, w, ci), device="cuda") * 0.1).to(
        torch.float8_e4m3fn
    )
    weight = (torch.randn((co, 3, 3, ci), device="cuda") * 0.1).to(
        torch.float8_e4m3fn
    )
    bias = (torch.randn(co, device="cuda") * 0.01).to(torch.bfloat16)
    out = torch.empty((n, h, w, co), device="cuda", dtype=torch.bfloat16)
    alpha = 0.75
    wrapper = lambda: source_call(
        ops,
        "fp8_conv2d_3x3_nhwc_bf16",
        input,
        weight,
        bias,
        alpha,
        out=out,
    )
    raw = lambda: native.conv2d(input, weight, bias, alpha, out)

    def cudnn_ref():
        x = input.float().permute(0, 3, 1, 2)
        wt = weight.float().permute(0, 3, 1, 2)
        y = F.conv2d(x, wt, padding=1).mul(alpha)
        y = y.add(bias.float().view(1, -1, 1, 1))
        out.copy_(y.permute(0, 2, 3, 1).to(torch.bfloat16))

    compiled = torch.compile(cudnn_ref, fullgraph=True)
    wrapper()
    got = out.clone()
    cudnn_ref()
    ref = out.clone()
    max_abs, mean_abs, p99_abs, cosine = metrics(got, ref)
    native_us = bench(raw, args.warmup, args.iters)
    wrapper_us = bench(wrapper, args.warmup, args.iters)
    eager_us = bench(cudnn_ref, args.warmup, args.iters)
    compile_us = bench(compiled, args.warmup, args.iters)
    return Result(
        label,
        str(shape),
        native_us,
        wrapper_us,
        wrapper_us / native_us,
        eager_us,
        compile_us,
        None,
        None,
        max_abs,
        mean_abs,
        p99_abs,
        cosine,
        wrapper_us - native_us <= max(0.5, native_us * 0.05)
        and wrapper_us <= min(eager_us, compile_us) * 0.98
        and cosine >= 0.999
        and mean_abs <= 0.01,
    )


def run_nvfp4_conv3d(ops, native, label, shape, args, residual_path):
    n, tc, tn, h, w, ci, co = shape
    cache_bf16 = (
        torch.randn((n, tc, h, w, ci), device="cuda") * 0.1
    ).to(torch.bfloat16)
    input_bf16 = (
        torch.randn((n, tn, h, w, ci), device="cuda") * 0.1
    ).to(torch.bfloat16)
    weight_bf16 = (
        torch.randn((co, 3, 3, 3, ci), device="cuda") * 0.1
    ).to(torch.bfloat16)
    cache, cache_sf = quantize_conv_tensor(cache_bf16)
    input, input_sf = quantize_conv_tensor(input_bf16)
    weight, weight_sf = quantize_conv_tensor(weight_bf16)
    cache_dequant = dequantize_linear_nvfp4(
        cache.reshape(-1, ci // 2), cache_sf.reshape(-1, ci // 16)
    ).reshape_as(cache_bf16)
    input_dequant = dequantize_linear_nvfp4(
        input.reshape(-1, ci // 2), input_sf.reshape(-1, ci // 16)
    ).reshape_as(input_bf16)
    weight_dequant = dequantize_linear_nvfp4(
        weight.reshape(-1, ci // 2), weight_sf.reshape(-1, ci // 16)
    ).reshape_as(weight_bf16)
    bias = (torch.randn(co, device="cuda") * 0.01).to(torch.bfloat16)
    alpha = 0.75

    if residual_path:
        residual = torch.randn(
            (n, co, tn, h, w), device="cuda", dtype=torch.bfloat16
        )
        out = torch.empty_like(residual)
        wrapper = lambda: source_call(
            ops,
            "nvfp4_causal_conv3d_residual_ncdhw_bf16",
            cache, input, weight, cache_sf, input_sf, weight_sf, bias,
            residual, None, alpha, out=out,
        )
        raw = lambda: native.nvfp4_causal_conv3d_residual(
            cache, input, weight, cache_sf, input_sf, weight_sf, bias,
            residual, alpha, out,
        )
    else:
        residual = None
        out = torch.empty(
            (n, tn, h, w, co), device="cuda", dtype=torch.bfloat16
        )
        wrapper = lambda: source_call(
            ops,
            "nvfp4_causal_conv3d_ndhwc_bf16",
            cache, input, weight, cache_sf, input_sf, weight_sf, bias,
            None, alpha, out=out,
        )
        raw = lambda: native.nvfp4_causal_conv3d(
            cache, input, weight, cache_sf, input_sf, weight_sf, bias,
            alpha, out,
        )

    def store_cudnn_result(cache_value, input_value):
        x = torch.cat((cache_value, input_value), dim=1).permute(
            0, 4, 1, 2, 3
        )
        wt = weight_dequant.permute(0, 4, 1, 2, 3)
        y = F.conv3d(x, wt, padding=(0, 1, 1))[:, :, :tn]
        y = y.mul(alpha).add(bias.float().view(1, -1, 1, 1, 1))
        if residual_path:
            out.copy_(
                (y.to(torch.bfloat16).float() + residual.float()).to(
                    torch.bfloat16
                )
            )
        else:
            out.copy_(y.permute(0, 2, 3, 4, 1).to(torch.bfloat16))

    def predequant_cudnn_ref():
        store_cudnn_result(cache_dequant, input_dequant)

    magnitude = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda"
    )
    scale_values, scale_bytes = [], []
    for byte in list(range(0x78)) + [0xFE]:
        exponent = (byte >> 3) & 0xF
        mantissa = byte & 0x7
        value = (
            (mantissa / 8.0) * (2.0 ** -6)
            if exponent == 0
            else (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))
        )
        scale_values.append(value)
        scale_bytes.append(byte)
    scale_lookup = torch.zeros(256, device="cuda")
    scale_lookup[
        torch.tensor(scale_bytes, device="cuda", dtype=torch.long)
    ] = torch.tensor(scale_values, device="cuda")

    def unpack(packed_value, scale_value):
        low = packed_value & 0xF
        high = packed_value >> 4
        low_value = magnitude[(low & 0x7).long()] * torch.where(
            low & 0x8 != 0, -1.0, 1.0
        )
        high_value = magnitude[(high & 0x7).long()] * torch.where(
            high & 0x8 != 0, -1.0, 1.0
        )
        values = torch.stack((low_value, high_value), dim=-1).flatten(-2)
        scales_value = scale_lookup[scale_value.long()].repeat_interleave(
            16, dim=-1
        )
        return values * scales_value

    def cudnn_ref():
        cache_value = unpack(cache, cache_sf).reshape_as(cache_bf16)
        input_value = unpack(input, input_sf).reshape_as(input_bf16)
        store_cudnn_result(cache_value, input_value)

    compiled = torch.compile(cudnn_ref, fullgraph=True)
    predequant_compiled = torch.compile(predequant_cudnn_ref, fullgraph=True)
    wrapper()
    got = out.clone()
    cudnn_ref()
    ref = out.clone()
    max_abs, mean_abs, p99_abs, cosine = metrics(got, ref)
    native_us = bench(raw, args.warmup, args.iters)
    wrapper_us = bench(wrapper, args.warmup, args.iters)
    eager_us = bench(cudnn_ref, args.warmup, args.iters)
    compile_us = bench(compiled, args.warmup, args.iters)
    diagnostic_eager_us = bench(
        predequant_cudnn_ref, args.warmup, args.iters
    )
    diagnostic_compile_us = bench(
        predequant_compiled, args.warmup, args.iters
    )
    return Result(
        f"{label}{'-residual' if residual_path else ''}",
        str(shape),
        native_us,
        wrapper_us,
        wrapper_us / native_us,
        eager_us,
        compile_us,
        diagnostic_eager_us,
        diagnostic_compile_us,
        max_abs,
        mean_abs,
        p99_abs,
        cosine,
        wrapper_us - native_us <= max(0.5, native_us * 0.05)
        and wrapper_us <= min(eager_us, compile_us) * 0.98
        and cosine >= 0.998
        and mean_abs <= 0.02,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--output")
    args = parser.parse_args()
    ops = (
        load_source_ops()
        if args.backend == "source"
        else load_installed_ops(args.artifact)
    )
    native = build_native()
    rows = [
        *(run_conv3d(ops, native, name, shape, args)
          for name, shape in CONV3D_SHAPES.items()),
        *(run_conv3d_residual(ops, native, name, shape, args)
          for name, shape in CONV3D_SHAPES.items()
          if shape[-1] % 8 == 0),
        *(run_conv2d(ops, native, name, shape, args)
          for name, shape in CONV2D_SHAPES.items()),
        *(run_nvfp4_conv3d(ops, native, name, shape, args, False)
          for name, shape in NVFP4_CONV3D_SHAPES.items()),
        *(run_nvfp4_conv3d(ops, native, name, shape, args, True)
          for name, shape in NVFP4_CONV3D_SHAPES.items()),
    ]
    for row in rows:
        print(
            f"{row.workload}: native={row.native_us:.3f}us "
            f"wrapper={row.wrapper_us:.3f}us ({row.wrapper_native:.3f}) "
            f"cuDNN-eager={row.eager_cudnn_us:.3f}us "
            f"cuDNN-compile={row.compile_cudnn_us:.3f}us "
            + (
                f"predequant-compile="
                f"{row.diagnostic_predequant_compile_us:.3f}us "
                if row.diagnostic_predequant_compile_us is not None
                else ""
            )
            + f"cos={row.cosine:.7f} accepted={row.accepted}"
        )
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(row) for row in rows], indent=2) + "\n")
    if not all(row.accepted for row in rows):
        raise SystemExit("world-model Conv acceptance failed")


if __name__ == "__main__":
    main()
