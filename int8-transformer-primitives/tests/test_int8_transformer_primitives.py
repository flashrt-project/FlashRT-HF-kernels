#!/usr/bin/env python3
"""Correctness tests for int8-transformer-primitives."""

from __future__ import annotations

import argparse
import importlib
import math
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "int8-transformer-primitives"
REGISTRATION_INCLUDE = ROOT.parent / "kernels" / "kernel-builder" / "src" / "pyproject" / "templates" / "torch"
CUTLASS_INCLUDE = ROOT.parent / "official" / "FlashRT" / "csrc" / "attention" / "flash_attn_2_src" / "cutlass" / "include"


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self.ops = getattr(torch.ops, namespace)

    def quantize_int8_static_bf16(self, x, scale):
        out = torch.empty_like(x, dtype=torch.int8)
        self.ops.quantize_int8_static_bf16(x, scale, out)
        return out

    def quantize_int8_rowwise_bf16(self, x):
        out = torch.empty_like(x, dtype=torch.int8)
        scales = torch.empty((x.shape[0],), device=x.device, dtype=torch.float32)
        self.ops.quantize_int8_rowwise_bf16(x, out, scales)
        return out, scales

    def quantize_int8_rowwise_static_bf16(self, x, scales):
        out = torch.empty_like(x, dtype=torch.int8)
        self.ops.quantize_int8_rowwise_static_bf16(x, scales, out)
        return out

    def rms_norm_quantize_int8_rowwise_bf16(self, x, weight, eps=1e-6, out=None, scales=None):
        if out is None:
            out = torch.empty_like(x, dtype=torch.int8)
        if scales is None:
            scales = torch.empty((x.shape[0],), device=x.device, dtype=torch.float32)
        self.ops.rms_norm_quantize_int8_rowwise_bf16(x, weight, float(eps), out, scales)
        return out, scales

    def residual_add_rms_norm_quantize_int8_rowwise_bf16(self, residual, x, weight, eps=1e-6, out=None, scales=None):
        if out is None:
            out = torch.empty_like(residual, dtype=torch.int8)
        if scales is None:
            scales = torch.empty((residual.shape[0],), device=residual.device, dtype=torch.float32)
        self.ops.residual_add_rms_norm_quantize_int8_rowwise_bf16(residual, x, weight, float(eps), out, scales)
        return out, scales

    def int8_rowwise_linear_bf16(self, x_i8, w_i8, x_scale, w_scale, variant=0, out=None):
        if out is None:
            out = torch.empty((x_i8.shape[0], w_i8.shape[0]), device=x_i8.device, dtype=torch.bfloat16)
        self.ops.int8_rowwise_linear_bf16(x_i8, w_i8, x_scale, w_scale, out, int(variant))
        return out

    def int8_silu_gated_linear_bf16(self, x_i8, w_i8, x_scale, w_scale, gate):
        out = torch.empty((x_i8.shape[0], w_i8.shape[0]), device=x_i8.device, dtype=torch.bfloat16)
        self.ops.int8_silu_gated_linear_bf16(x_i8, w_i8, x_scale, w_scale, gate, out)
        return out


def _arch_list() -> str:
    major, minor = torch.cuda.get_device_capability(0)
    return "12.0a" if major >= 12 else f"{major}.{minor}"


def load_source_ops() -> SourceOps:
    from torch.utils.cpp_extension import load

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _arch_list())
    namespace = "int8_transformer_primitives_source_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "int8_transformer_primitives.cu"),
            str(PACKAGE / "csrc" / "gemm" / "cutlass_sm80_int8_rowwise.cu"),
            str(PACKAGE / "csrc" / "gemm" / "cutlass_sm80_int8_rowwise_t64x128.cu"),
            str(PACKAGE / "csrc" / "gemm" / "cutlass_sm80_int8_silu_gated.cu"),
        ],
        extra_include_paths=[str(PACKAGE / "csrc"), str(CUTLASS_INCLUDE), str(REGISTRATION_INCLUDE)],
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
        return importlib.import_module("int8_transformer_primitives")
    finally:
        if artifact:
            sys.path.remove(artifact)


def quant_ref(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.round(x.float() * (1.0 / scale.float())), -127, 127).to(torch.int8)


def rowwise_ref(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scales = torch.clamp(x.float().abs().amax(dim=1) / 127.0, min=1e-10)
    return quant_ref(x, scales[:, None]), scales


def rms_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rms = torch.rsqrt((x.float() * x.float()).mean(dim=1, keepdim=True) + eps)
    return x.float() * rms * weight.float()


def metrics(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float]:
    diff = (got.float() - ref.float()).abs()
    cos = torch.nn.functional.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0).item()
    return float(diff.max().item()), float(diff.mean().item()), float(cos)


def assert_close(name: str, got: torch.Tensor, ref: torch.Tensor, atol: float, cos_min: float) -> None:
    max_abs, mean_abs, cos = metrics(got, ref)
    print(f"{name}: max_abs={max_abs:.6f} mean_abs={mean_abs:.6f} cosine={cos:.8f}")
    if max_abs > atol or cos < cos_min:
        raise AssertionError(f"{name} failed: max_abs={max_abs} cosine={cos}")


def assert_quantized_equal_or_tie(name: str, got: torch.Tensor, ref: torch.Tensor, values: torch.Tensor, scales: torch.Tensor) -> None:
    diff = (got.int() - ref.int()).abs()
    bad = diff != 0
    if not bool(bad.any().item()):
        return
    scaled = values.float() * (1.0 / scales.float())
    frac_dist = (scaled.abs() - torch.floor(scaled.abs()) - 0.5).abs()
    tie = frac_dist < 2e-4
    invalid = bad & (~tie | (diff > 1))
    print(
        f"{name}: tie_bin_count={int((bad & tie).sum().item())} "
        f"invalid_count={int(invalid.sum().item())}"
    )
    if bool(invalid.any().item()):
        idx = invalid.nonzero()[0].tolist()
        raise AssertionError(
            f"{name} mismatch outside quantization tie boundary at {idx}: "
            f"got={got[tuple(idx)].item()} ref={ref[tuple(idx)].item()}"
        )


def run(ops, mode: str) -> int:
    torch.manual_seed(23)
    shapes = [(4, 128), (17, 1024)] if mode == "smoke" else [(1, 128), (4, 128), (17, 1024), (64, 2048), (257, 2048)]
    count = 0
    for rows, cols in shapes:
        x = (torch.randn((rows, cols), device="cuda") * 0.5).to(torch.bfloat16)
        scale = torch.tensor([0.05], device="cuda", dtype=torch.float32)
        got = ops.quantize_int8_static_bf16(x, scale)
        ref = quant_ref(x, scale)
        if not torch.equal(got.cpu(), ref.cpu()):
            raise AssertionError(f"static quant mismatch rows={rows} cols={cols}")
        count += 1

        got, scales = ops.quantize_int8_rowwise_bf16(x)
        ref, ref_scales = rowwise_ref(x)
        assert_quantized_equal_or_tie(f"rowwise quant rows={rows} cols={cols}", got, ref, x, scales[:, None])
        torch.testing.assert_close(scales.cpu(), ref_scales.cpu(), rtol=1e-6, atol=1e-8)
        got_static = ops.quantize_int8_rowwise_static_bf16(x, scales)
        assert_quantized_equal_or_tie("rowwise static quant", got_static, got, x, scales[:, None])
        count += 2

        weight = torch.randn((cols,), device="cuda", dtype=torch.bfloat16)
        got, scales = ops.rms_norm_quantize_int8_rowwise_bf16(x, weight)
        ref_norm = rms_ref(x, weight)
        ref_i8, ref_scales = rowwise_ref(ref_norm)
        assert_quantized_equal_or_tie(f"rms norm quant rows={rows} cols={cols}", got, ref_i8, ref_norm, scales[:, None])
        torch.testing.assert_close(scales.cpu(), ref_scales.cpu(), rtol=2e-3, atol=1e-5)
        count += 1

        residual = torch.randn_like(x)
        residual_sum = residual.float() + x.float()
        residual_ref = residual_sum.to(torch.bfloat16)
        residual_mut = residual.clone()
        got, _ = ops.residual_add_rms_norm_quantize_int8_rowwise_bf16(residual_mut, x, weight)
        torch.testing.assert_close(residual_mut.float().cpu(), residual_ref.float().cpu(), rtol=0, atol=0)
        ref_i8, _ = rowwise_ref(rms_ref(residual_sum, weight))
        assert_quantized_equal_or_tie(
            f"residual rms quant rows={rows} cols={cols}", got, ref_i8, rms_ref(residual_sum, weight), scales[:, None]
        )
        count += 1

    gemm_shapes = [(8, 128, 128), (17, 256, 256)] if mode == "smoke" else [
        (1, 128, 128),
        (8, 128, 128),
        (17, 256, 256),
        (64, 512, 1024),
        (257, 2048, 2560),
    ]
    for m, k, n in gemm_shapes:
        x = (torch.randn((m, k), device="cuda") * 0.5).to(torch.bfloat16)
        w = (torch.randn((n, k), device="cuda") * 0.5).to(torch.bfloat16)
        x_i8, x_scale = ops.quantize_int8_rowwise_bf16(x)
        w_i8, w_scale = ops.quantize_int8_rowwise_bf16(w)
        got = ops.int8_rowwise_linear_bf16(x_i8, w_i8, x_scale, w_scale)
        ref = ((x_i8.float() @ w_i8.float().t()) * x_scale[:, None] * w_scale[None, :]).to(torch.bfloat16)
        assert_close(f"int8_linear m={m} k={k} n={n}", got, ref, atol=0.25, cos_min=0.999)
        count += 1

        if n <= 1024:
            gate = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)
            got = ops.int8_silu_gated_linear_bf16(x_i8, w_i8, x_scale, w_scale, gate)
            ref = (torch.nn.functional.silu(gate.float()) * ref.float()).to(torch.bfloat16)
            assert_close(f"int8_silu_gated m={m} k={k} n={n}", got, ref, atol=0.25, cos_min=0.999)
            count += 1

    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    count = run(ops, args.mode)
    print(f"int8-transformer-primitives {args.backend} {args.mode}: passed {count}/{count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
