#!/usr/bin/env python3
"""Correctness tests for flashrt-gemm-epilogues FP8 quantization epilogues."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "flashrt-gemm-epilogues"
REGISTRATION_INCLUDE = (
    ROOT.parent
    / "kernels"
    / "kernel-builder"
    / "src"
    / "pyproject"
    / "templates"
    / "torch"
)

_SOURCE_LIST = [
    "torch-ext/torch_binding.cpp",
    "csrc/bf16_gemm_bias_gelu.cu",
    "csrc/bias_gelu_quantize_fp8.cu",
    "csrc/channel_scale_quantize_fp8.cu",
]


def _fp8_dtype():
    if torch.version.hip is not None and hasattr(torch, "float8_e4m3fnuz"):
        return torch.float8_e4m3fnuz
    return torch.float8_e4m3fn


def _fp8_max() -> float:
    return 240.0 if torch.version.hip is not None else 448.0


def _arch_list() -> str:
    major, minor = torch.cuda.get_device_capability(0)
    if major >= 12:
        return "12.0a"
    if (major, minor) == (11, 0):
        return "11.0a"
    return f"{major}.{minor}"


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self._ops = getattr(torch.ops, namespace)

    def bias_gelu_quantize(self, input, bias, scale, *, out=None):
        if out is None:
            out = torch.empty(input.shape, device=input.device, dtype=_fp8_dtype())
        self._ops.bias_gelu_quantize_fp8_static_bf16(input, bias, scale, out)
        return out

    def gelu_quantize(self, input, scale, *, out=None):
        if out is None:
            out = torch.empty(input.shape, device=input.device, dtype=_fp8_dtype())
        self._ops.gelu_quantize_fp8_static_bf16(input, scale, out)
        return out

    def channel_scale_quantize(self, input, channel_scale, scale, *, out=None):
        if out is None:
            out = torch.empty(input.shape, device=input.device, dtype=_fp8_dtype())
        self._ops.channel_scale_quantize_fp8_static_bf16(input, channel_scale, scale, out)
        return out


class InstalledOps:
    def __init__(self, module) -> None:
        self._module = module

    def bias_gelu_quantize(self, input, bias, scale, *, out=None):
        return self._module.bias_gelu_quantize_fp8_static_bf16(input, bias, scale, out=out)

    def gelu_quantize(self, input, scale, *, out=None):
        return self._module.gelu_quantize_fp8_static_bf16(input, scale, out=out)

    def channel_scale_quantize(self, input, channel_scale, scale, *, out=None):
        return self._module.channel_scale_quantize_fp8_static_bf16(
            input, channel_scale, scale, out=out
        )


def _preload_cublaslt() -> None:
    import ctypes
    import ctypes.util

    for parent in Path(torch.__file__).resolve().parents:
        candidate = parent / "nvidia" / "cublas" / "lib" / "libcublasLt.so.12"
        if candidate.exists():
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
            return
    library = ctypes.util.find_library("cublasLt")
    if library:
        ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)


def load_source_ops() -> SourceOps:
    from torch.utils.cpp_extension import load

    if not REGISTRATION_INCLUDE.is_dir():
        raise RuntimeError(f"missing kernel-builder registration include: {REGISTRATION_INCLUDE}")
    _preload_cublaslt()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _arch_list())
    namespace = "flashrt_gemm_epilogues_test"
    load(
        name=namespace,
        sources=[str(PACKAGE / s) for s in _SOURCE_LIST],
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
        return InstalledOps(importlib.import_module("flashrt_gemm_epilogues"))
    finally:
        if artifact:
            sys.path.remove(artifact)


def _reference(input, bias, scale):
    y = input.float()
    if bias is not None:
        y = y + bias.float()
    y = torch.nn.functional.gelu(y, approximate="tanh")
    y = torch.clamp(y / scale.float(), -_fp8_max(), _fp8_max())
    return y.to(_fp8_dtype())


def _channel_reference(input, channel_scale, scale):
    y = input.float() * channel_scale.float()
    y = torch.clamp(y / scale.float(), -_fp8_max(), _fp8_max())
    return y.to(_fp8_dtype())


def _check_bias_gelu(ops, shape) -> None:
    torch.manual_seed(0)
    input = torch.randn(shape, device="cuda", dtype=torch.bfloat16).contiguous()
    bias = torch.randn((shape[-1],), device="cuda", dtype=torch.bfloat16).contiguous()
    scale = torch.tensor([0.25], device="cuda", dtype=torch.float32)
    out = ops.bias_gelu_quantize(input, bias, scale)
    expected = _reference(input, bias, scale)
    torch.testing.assert_close(out.float(), expected.float(), rtol=0, atol=0)
    print(f"PASS bias_gelu_quantize_fp8 shape={shape}: exact")


def _check_gelu(ops) -> None:
    torch.manual_seed(1)
    input = torch.randn((8, 64), device="cuda", dtype=torch.bfloat16).contiguous()
    scale = torch.tensor([0.5], device="cuda", dtype=torch.float32)
    out = ops.gelu_quantize(input, scale)
    expected = _reference(input, None, scale)
    torch.testing.assert_close(out.float(), expected.float(), rtol=0, atol=0)
    print("PASS gelu_quantize_fp8 (no bias): exact")


def _check_channel_scale(ops, shape) -> None:
    torch.manual_seed(2)
    input = torch.randn(shape, device="cuda", dtype=torch.bfloat16).contiguous()
    channel_scale = torch.randn((shape[-1],), device="cuda", dtype=torch.bfloat16).contiguous()
    scale = torch.tensor([0.25], device="cuda", dtype=torch.float32)
    out = ops.channel_scale_quantize(input, channel_scale, scale)
    expected = _channel_reference(input, channel_scale, scale)
    torch.testing.assert_close(out.float(), expected.float(), rtol=0, atol=0)
    print(f"PASS channel_scale_quantize_fp8 shape={shape}: exact")


def _check_reuse_and_reject(ops) -> None:
    input = torch.randn((2, 16), device="cuda", dtype=torch.bfloat16).contiguous()
    bias = torch.randn((16,), device="cuda", dtype=torch.bfloat16).contiguous()
    scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    out = torch.empty(input.shape, device="cuda", dtype=_fp8_dtype())
    returned = ops.bias_gelu_quantize(input, bias, scale, out=out)
    if returned is not out:
        raise AssertionError("out tensor must be reused in place")
    for fn_args, match in [
        ((input, torch.randn((15,), device="cuda", dtype=torch.bfloat16).contiguous(), scale), "bias length"),
        ((input, torch.randn((15,), device="cuda", dtype=torch.bfloat16).contiguous(), scale), "channel_scale length"),
    ]:
        if match == "bias length":
            try:
                ops.bias_gelu_quantize(*fn_args)
            except RuntimeError as exc:
                if match not in str(exc):
                    raise
            else:
                raise AssertionError("wrong bias shape must be rejected")
        else:
            try:
                ops.channel_scale_quantize(input, fn_args[1], scale)
            except RuntimeError as exc:
                if match not in str(exc):
                    raise
            else:
                raise AssertionError("wrong channel_scale shape must be rejected")
    print("PASS fp8 epilogue reuse + rejections")


def run(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not (hasattr(torch, "float8_e4m3fn") or hasattr(torch, "float8_e4m3fnuz")):
        raise RuntimeError("FP8 support is required")
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    shapes = [(4, 16)] if args.mode == "smoke" else [(4, 16), (2, 3, 32)]
    for shape in shapes:
        _check_bias_gelu(ops, shape)
        _check_channel_scale(ops, shape)
    _check_gelu(ops)
    _check_reuse_and_reject(ops)
    print(f"PASS flashrt-gemm-epilogues/fp8-quant {args.backend} mode={args.mode}: "
          f"{2 * len(shapes) + 2} checks")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        import traceback
        traceback.print_exc()
        return 1
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps({"passed": 1, "total": 1, "backend": args.backend}) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
