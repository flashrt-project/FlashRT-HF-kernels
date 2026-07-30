#!/usr/bin/env python3
"""Correctness tests for world-model-conv."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import importlib
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "world-model-conv"
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

    def fp8_conv3d_v18_ncdhw_res_bf16out(self, cache_x, new_x, weight, bias, residual, alpha=1.0):
        out = torch.empty_like(residual)
        self._ops.fp8_conv3d_v18_ncdhw_res_bf16out(cache_x, new_x, weight, bias, residual, float(alpha), out)
        return out

    def fp8_causal_conv3d_ndhwc_bf16(
        self, cache_x, new_x, weight, bias, alpha=1.0
    ):
        out = torch.empty(
            (*new_x.shape[:4], weight.shape[0]),
            device=new_x.device,
            dtype=torch.bfloat16,
        )
        self._ops.fp8_causal_conv3d_ndhwc_bf16(
            cache_x, new_x, weight, bias, float(alpha), out
        )
        return out

    def fp8_conv2d_3x3_nhwc_bf16(
        self, input, weight, bias, alpha=1.0
    ):
        out = torch.empty(
            (*input.shape[:3], weight.shape[0]),
            device=input.device,
            dtype=torch.bfloat16,
        )
        self._ops.fp8_conv2d_3x3_nhwc_bf16(
            input, weight, bias, float(alpha), out
        )
        return out

    def fp8_conv2d_3x3_ncdhw_bf16(
        self, input, weight, bias, alpha=1.0
    ):
        out = torch.empty(
            (
                input.shape[0],
                weight.shape[0],
                input.shape[1],
                input.shape[2],
                input.shape[3],
            ),
            device=input.device,
            dtype=torch.bfloat16,
        )
        self._ops.fp8_conv2d_3x3_ncdhw_bf16(
            input, weight, bias, float(alpha), out
        )
        return out

    def nvfp4_causal_conv3d_ndhwc_bf16(
        self, cache, new, weight, cache_sf, new_sf, weight_sf, bias,
        outer_weight=None, alpha=1.0
    ):
        out = torch.empty(
            (*new.shape[:4], weight.shape[0]),
            device=new.device,
            dtype=torch.bfloat16,
        )
        self._ops.nvfp4_causal_conv3d_ndhwc_bf16(
            cache, new, weight, cache_sf, new_sf, weight_sf, bias,
            outer_weight, float(alpha), out
        )
        return out

    def nvfp4_causal_conv3d_residual_ncdhw_bf16(
        self, cache, new, weight, cache_sf, new_sf, weight_sf, bias,
        residual, outer_weight=None, alpha=1.0
    ):
        out = torch.empty_like(residual)
        self._ops.nvfp4_causal_conv3d_residual_ncdhw_bf16(
            cache, new, weight, cache_sf, new_sf, weight_sf, bias,
            residual, outer_weight, float(alpha), out
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
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0a"
    namespace = "world_model_conv_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "fp8_conv3d_sm120_v18.cu"),
            str(PACKAGE / "csrc" / "fp8_causal_conv3d_sm120.cu"),
            str(PACKAGE / "csrc" / "fp8_conv2d_3x3_sm120.cu"),
            str(PACKAGE / "csrc" / "nvfp4_causal_conv3d_sm120.cu"),
            str(PACKAGE / "csrc" / "nvfp4_causal_conv3d_residual_sm120.cu"),
            str(PACKAGE / "csrc" / "nvfp4_causal_conv3d_residual_k128_sm120.cu"),
        ],
        extra_include_paths=[str(PACKAGE / "csrc"), str(REGISTRATION_INCLUDE)],
        extra_cflags=["-O3", "-DCUDA_KERNEL"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-DCUDA_KERNEL"],
        is_python_module=False,
        verbose=False,
    )
    return SourceOps(namespace)


def load_installed_ops(artifact: str | None):
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return importlib.import_module("world_model_conv")
    finally:
        if artifact:
            sys.path.remove(artifact)


def ref_conv(cache_x, new_x, weight, bias, residual, alpha: float) -> torch.Tensor:
    x = torch.cat([cache_x, new_x], dim=1).to(torch.float32)
    # NDHWC -> NCDHW
    x_ncdhw = x.permute(0, 4, 1, 2, 3).contiguous()
    # (Co,3,3,3,Ci) -> (Co,Ci,3,3,3)
    w = weight.to(torch.float32).permute(0, 4, 1, 2, 3).contiguous()
    y = F.conv3d(x_ncdhw, w, bias=None, stride=1, padding=(0, 1, 1))
    y = y * float(alpha) + bias.float().view(1, -1, 1, 1, 1)
    y_bf16 = y.to(torch.bfloat16)
    return (y_bf16.float() + residual.float()).to(torch.bfloat16)


def ref_causal_conv3d(cache_x, new_x, weight, bias, alpha: float) -> torch.Tensor:
    x = torch.cat([cache_x, new_x], dim=1).to(torch.float32)
    x_ncdhw = x.permute(0, 4, 1, 2, 3).contiguous()
    w = weight.to(torch.float32).permute(0, 4, 1, 2, 3).contiguous()
    y = F.conv3d(x_ncdhw, w, padding=(0, 1, 1))
    y = y * float(alpha) + bias.float().view(1, -1, 1, 1, 1)
    return y[:, :, : new_x.shape[1]].permute(0, 2, 3, 4, 1).to(
        torch.bfloat16
    )


def ref_conv2d(input, weight, bias, alpha: float) -> torch.Tensor:
    x = input.float().permute(0, 3, 1, 2).contiguous()
    w = weight.float().permute(0, 3, 1, 2).contiguous()
    y = F.conv2d(x, w, padding=1) * float(alpha)
    y = y + bias.float().view(1, -1, 1, 1)
    return y.permute(0, 2, 3, 1).to(torch.bfloat16)


def _ue4m3_table(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = list(range(0x78)) + [0xFE]
    values = []
    for byte in encoded:
        exponent = (byte >> 3) & 0xF
        mantissa = byte & 0x7
        if exponent == 0:
            value = (mantissa / 8.0) * (2.0 ** -6)
        else:
            value = (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))
        values.append(value)
    return (
        torch.tensor(values, device=device, dtype=torch.float32),
        torch.tensor(encoded, device=device, dtype=torch.uint8),
    )


def quantize_linear_nvfp4(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows = input.reshape(-1, input.shape[-1]).float()
    cols = rows.shape[1]
    blocks = rows.reshape(rows.shape[0], cols // 16, 16)
    required_scale = blocks.abs().amax(dim=-1) / 6.0
    scale_values, scale_bytes = _ue4m3_table(input.device)
    scale_index = torch.searchsorted(scale_values, required_scale).clamp(
        max=scale_values.numel() - 1
    )
    scales = scale_values[scale_index]
    encoded_scales = scale_bytes[scale_index]
    normalized = blocks / torch.where(
        scales > 0.0, scales, torch.ones_like(scales)
    ).unsqueeze(-1)
    thresholds = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        device=input.device,
    )
    magnitude = (
        normalized.abs().unsqueeze(-1) >= thresholds
    ).sum(dim=-1).to(torch.uint8)
    encoded = magnitude | ((normalized < 0.0).to(torch.uint8) << 3)
    encoded = encoded.reshape(rows.shape[0], cols)
    packed = encoded[:, 0::2] | (encoded[:, 1::2] << 4)
    return packed.contiguous(), encoded_scales.contiguous()


def dequantize_linear_nvfp4(
    packed: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    magnitude = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=packed.device,
    )
    low = packed & 0xF
    high = packed >> 4
    low_value = magnitude[(low & 0x7).long()] * torch.where(
        low & 0x8 != 0, -1.0, 1.0
    )
    high_value = magnitude[(high & 0x7).long()] * torch.where(
        high & 0x8 != 0, -1.0, 1.0
    )
    values = torch.stack((low_value, high_value), dim=-1).flatten(-2)
    scale_values, scale_bytes = _ue4m3_table(packed.device)
    lookup = torch.zeros(256, device=packed.device)
    lookup[scale_bytes.long()] = scale_values
    decoded_scales = lookup[scales.long()]
    return values * decoded_scales.repeat_interleave(16, dim=-1)


def quantize_conv_tensor(input: torch.Tensor):
    packed, scales = quantize_linear_nvfp4(input)
    return (
        packed.reshape(*input.shape[:-1], input.shape[-1] // 2),
        scales.reshape(*input.shape[:-1], input.shape[-1] // 16),
    )


def assert_close(name: str, got: torch.Tensor, ref: torch.Tensor, max_atol: float, mean_atol: float, min_cos: float) -> None:
    diff = (got.float() - ref.float()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    cos = torch.nn.functional.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0).item()
    if max_err > max_atol or mean_err > mean_atol or cos < min_cos:
        raise AssertionError(f"{name}: max_err={max_err:.8f}, mean_err={mean_err:.8f}, cos={cos:.8f}")


def run_tests(ops) -> int:
    count = 0
    shapes = [
        (1, 2, 1, 8, 8, 32, 16),
        (1, 2, 4, 16, 16, 32, 32),
        (2, 2, 4, 16, 24, 64, 32),
    ]
    for n, tc, tn, h, w, ci, co in shapes:
        cache = (torch.randn((n, tc, h, w, ci), device="cuda") * 0.1).to(torch.float8_e4m3fn)
        new = (torch.randn((n, tn, h, w, ci), device="cuda") * 0.1).to(torch.float8_e4m3fn)
        weight = (torch.randn((co, 3, 3, 3, ci), device="cuda") * 0.1).to(torch.float8_e4m3fn)
        bias = (torch.randn((co,), device="cuda") * 0.01).to(torch.bfloat16)
        residual = (torch.randn((n, co, tn, h, w), device="cuda") * 0.05).to(torch.bfloat16)
        alpha = 0.75
        got = ops.fp8_conv3d_v18_ncdhw_res_bf16out(cache, new, weight, bias, residual, alpha)
        ref = ref_conv(cache, new, weight, bias, residual, alpha)
        assert_close(f"fp8_conv3d shape={(n,tc,tn,h,w,ci,co)}", got, ref, 0.125, 0.01, 0.999)
        count += 1

    conv3d_shapes = [
        (1, 2, 1, 8, 8, 32, 16),
        (1, 2, 4, 16, 16, 64, 32),
        (1, 2, 3, 9, 7, 32, 18),
    ]
    for n, tc, tn, h, w, ci, co in conv3d_shapes:
        cache = (torch.randn((n, tc, h, w, ci), device="cuda") * 0.1).to(
            torch.float8_e4m3fn
        )
        new = (torch.randn((n, tn, h, w, ci), device="cuda") * 0.1).to(
            torch.float8_e4m3fn
        )
        weight = (
            torch.randn((co, 3, 3, 3, ci), device="cuda") * 0.1
        ).to(torch.float8_e4m3fn)
        bias = (torch.randn(co, device="cuda") * 0.01).to(torch.bfloat16)
        got = ops.fp8_causal_conv3d_ndhwc_bf16(
            cache, new, weight, bias, 0.75
        )
        ref = ref_causal_conv3d(cache, new, weight, bias, 0.75)
        assert_close(
            f"fp8_causal_conv3d shape={(n,tc,tn,h,w,ci,co)}",
            got,
            ref,
            0.125,
            0.01,
            0.999,
        )
        count += 1

    conv2d_shapes = [
        (1, 8, 8, 32, 16),
        (4, 16, 16, 64, 32),
        (17, 9, 7, 32, 24),
    ]
    for n, h, w, ci, co in conv2d_shapes:
        input = (torch.randn((n, h, w, ci), device="cuda") * 0.1).to(
            torch.float8_e4m3fn
        )
        weight = (torch.randn((co, 3, 3, ci), device="cuda") * 0.1).to(
            torch.float8_e4m3fn
        )
        bias = (torch.randn(co, device="cuda") * 0.01).to(torch.bfloat16)
        got = ops.fp8_conv2d_3x3_nhwc_bf16(
            input, weight, bias, 0.75
        )
        ref = ref_conv2d(input, weight, bias, 0.75)
        assert_close(
            f"fp8_conv2d shape={(n,h,w,ci,co)}",
            got,
            ref,
            0.125,
            0.01,
            0.999,
        )
        bt = input.reshape(1, n, h, w, ci)
        got_ncdhw = ops.fp8_conv2d_3x3_ncdhw_bf16(
            bt, weight, bias, 0.75
        )
        ref_ncdhw = ref.reshape(1, n, h, w, co).permute(
            0, 4, 1, 2, 3
        ).contiguous()
        assert_close(
            f"fp8_conv2d_ncdhw shape={(1,n,h,w,ci,co)}",
            got_ncdhw,
            ref_ncdhw,
            0.125,
            0.01,
            0.999,
        )
        count += 2

    unsupported_cache = torch.empty(
        (1, 2, 4, 4, 128), device="cuda", dtype=torch.float8_e4m3fn
    )
    unsupported_new = torch.empty_like(unsupported_cache)
    unsupported_weight = torch.empty(
        (16, 3, 3, 3, 128),
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    unsupported_bias = torch.empty(
        16, device="cuda", dtype=torch.bfloat16
    )
    try:
        ops.fp8_causal_conv3d_ndhwc_bf16(
            unsupported_cache,
            unsupported_new,
            unsupported_weight,
            unsupported_bias,
            1.0,
        )
        raise AssertionError("FP8 Conv3D Ci=128 must be rejected")
    except RuntimeError as exc:
        if "Ci=32/64" not in str(exc):
            raise
    count += 1

    nvfp4_shapes = [
        (1, 2, 1, 8, 8, 64, 16),
        (1, 2, 3, 9, 7, 128, 16),
        (1, 2, 2, 8, 8, 512, 16),
    ]
    for n, tc, tn, h, w, ci, co in nvfp4_shapes:
        cache_bf16 = (
            torch.randn((n, tc, h, w, ci), device="cuda") * 0.1
        ).to(torch.bfloat16)
        new_bf16 = (
            torch.randn((n, tn, h, w, ci), device="cuda") * 0.1
        ).to(torch.bfloat16)
        weight_bf16 = (
            torch.randn((co, 3, 3, 3, ci), device="cuda") * 0.1
        ).to(torch.bfloat16)
        cache, cache_sf = quantize_conv_tensor(cache_bf16)
        new, new_sf = quantize_conv_tensor(new_bf16)
        weight, weight_sf = quantize_conv_tensor(weight_bf16)
        cache_dequant = dequantize_linear_nvfp4(
            cache.reshape(-1, ci // 2), cache_sf.reshape(-1, ci // 16)
        ).reshape_as(cache_bf16)
        new_dequant = dequantize_linear_nvfp4(
            new.reshape(-1, ci // 2), new_sf.reshape(-1, ci // 16)
        ).reshape_as(new_bf16)
        weight_dequant = dequantize_linear_nvfp4(
            weight.reshape(-1, ci // 2), weight_sf.reshape(-1, ci // 16)
        ).reshape_as(weight_bf16)
        bias = (torch.randn(co, device="cuda") * 0.01).to(torch.bfloat16)
        alpha = 0.75
        got = ops.nvfp4_causal_conv3d_ndhwc_bf16(
            cache, new, weight, cache_sf, new_sf, weight_sf, bias,
            None, alpha
        )
        ref = ref_causal_conv3d(
            cache_dequant, new_dequant, weight_dequant, bias, alpha
        )
        assert_close(
            f"nvfp4_causal_conv3d shape={(n,tc,tn,h,w,ci,co)}",
            got,
            ref,
            0.25,
            0.02,
            0.998,
        )
        count += 1

        residual = (
            torch.randn((n, co, tn, h, w), device="cuda") * 0.05
        ).to(torch.bfloat16)
        got_residual = ops.nvfp4_causal_conv3d_residual_ncdhw_bf16(
            cache, new, weight, cache_sf, new_sf, weight_sf, bias,
            residual, None, alpha
        )
        ref_residual = (
            ref.permute(0, 4, 1, 2, 3).float() + residual.float()
        ).to(torch.bfloat16)
        torch.testing.assert_close(
            got_residual, ref_residual, rtol=0.0, atol=0.0
        )
        count += 1

        outer_weight = (
            torch.rand(co, device="cuda", dtype=torch.float32) * 0.5 + 0.75
        )
        got_outer = ops.nvfp4_causal_conv3d_ndhwc_bf16(
            cache, new, weight, cache_sf, new_sf, weight_sf, bias,
            outer_weight, alpha
        )
        conv = F.conv3d(
            torch.cat([cache_dequant, new_dequant], dim=1)
            .permute(0, 4, 1, 2, 3),
            weight_dequant.permute(0, 4, 1, 2, 3),
            padding=(0, 1, 1),
        )[:, :, :tn]
        ref_outer = (
            conv
            * (outer_weight * alpha).view(1, -1, 1, 1, 1)
            + bias.float().view(1, -1, 1, 1, 1)
        ).permute(0, 2, 3, 4, 1).to(torch.bfloat16)
        assert_close(
            f"nvfp4_causal_conv3d_outer shape={(n,tc,tn,h,w,ci,co)}",
            got_outer,
            ref_outer,
            0.25,
            0.02,
            0.998,
        )
        count += 1

    ci = 320
    co = 16
    cache_bf16 = torch.randn(
        (1, 2, 4, 4, ci), device="cuda", dtype=torch.bfloat16
    )
    new_bf16 = torch.randn(
        (1, 1, 4, 4, ci), device="cuda", dtype=torch.bfloat16
    )
    weight_bf16 = torch.randn(
        (co, 3, 3, 3, ci), device="cuda", dtype=torch.bfloat16
    )
    cache, cache_sf = quantize_conv_tensor(cache_bf16)
    new, new_sf = quantize_conv_tensor(new_bf16)
    weight, weight_sf = quantize_conv_tensor(weight_bf16)
    try:
        ops.nvfp4_causal_conv3d_ndhwc_bf16(
            cache, new, weight, cache_sf, new_sf, weight_sf,
            torch.zeros(co, device="cuda", dtype=torch.bfloat16),
            None, 1.0,
        )
        raise AssertionError("NVFP4 Conv3D Ci=320 must be rejected")
    except RuntimeError as exc:
        if "Ci=64 or multiples of 128" not in str(exc):
            raise
    count += 1
    return count


def run_compile_tests(ops) -> int:
    cache = (torch.randn((1, 2, 8, 8, 32), device="cuda") * 0.1).to(
        torch.float8_e4m3fn
    )
    new = (torch.randn((1, 2, 8, 8, 32), device="cuda") * 0.1).to(
        torch.float8_e4m3fn
    )
    weight3d = (torch.randn((16, 3, 3, 3, 32), device="cuda") * 0.1).to(
        torch.float8_e4m3fn
    )
    weight2d = (torch.randn((16, 3, 3, 32), device="cuda") * 0.1).to(
        torch.float8_e4m3fn
    )
    bias = torch.randn(16, device="cuda", dtype=torch.bfloat16)

    def conv3d(c, x, w, b):
        return ops.fp8_causal_conv3d_ndhwc_bf16(c, x, w, b, 0.75)

    def conv2d(x, w, b):
        return ops.fp8_conv2d_3x3_nhwc_bf16(x, w, b, 0.75)

    for name, fn, values in [
        ("fp8_causal_conv3d", conv3d, (cache, new, weight3d, bias)),
        ("fp8_conv2d", conv2d, (new.reshape(2, 8, 8, 32), weight2d, bias)),
    ]:
        eager = fn(*values)
        compiled = torch.compile(fn, fullgraph=True)(*values)
        torch.testing.assert_close(compiled, eager, rtol=0.0, atol=0.0)
        print(f"PASS {name} torch.compile fullgraph")
    cache_bf16 = torch.randn(
        (1, 2, 8, 8, 64), device="cuda", dtype=torch.bfloat16
    )
    new_bf16 = torch.randn(
        (1, 2, 8, 8, 64), device="cuda", dtype=torch.bfloat16
    )
    weight_bf16 = torch.randn(
        (16, 3, 3, 3, 64), device="cuda", dtype=torch.bfloat16
    )
    cache4, cache_sf = quantize_conv_tensor(cache_bf16)
    new4, new_sf = quantize_conv_tensor(new_bf16)
    weight4, weight_sf = quantize_conv_tensor(weight_bf16)

    def nvfp4_conv(c, x, w, cs, xs, ws, b):
        return ops.nvfp4_causal_conv3d_ndhwc_bf16(
            c, x, w, cs, xs, ws, b, None, 0.75
        )

    eager = nvfp4_conv(
        cache4, new4, weight4, cache_sf, new_sf, weight_sf, bias
    )
    compiled = torch.compile(nvfp4_conv, fullgraph=True)(
        cache4, new4, weight4, cache_sf, new_sf, weight_sf, bias
    )
    torch.testing.assert_close(compiled, eager, rtol=0.0, atol=0.0)
    print("PASS nvfp4_causal_conv3d torch.compile fullgraph")
    return 3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    major, _ = torch.cuda.get_device_capability(0)
    if major < 12:
        raise RuntimeError("world-model-conv source validation requires Blackwell SM120+")
    torch.manual_seed(0)
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    total = run_tests(ops)
    total += run_compile_tests(ops)
    torch.cuda.synchronize()
    print(f"world-model-conv correctness passed: {total} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
