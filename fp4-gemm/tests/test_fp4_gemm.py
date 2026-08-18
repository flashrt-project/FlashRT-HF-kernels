#!/usr/bin/env python3
"""Correctness tests for fp4-gemm."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "fp4-gemm"
REGISTRATION_INCLUDE = (
    ROOT.parent
    / "kernels"
    / "kernel-builder"
    / "src"
    / "pyproject"
    / "templates"
    / "torch"
)
DEFAULT_CUTLASS_INCLUDE = (
    ROOT.parent
    / "flashrt_pr31_review"
    / "third_party"
    / "cutlass"
    / "include"
)

SHAPES = {
    "small_m16_n128_k128": (16, 128, 128),
    "small_m32_n256_k256": (32, 256, 256),
    "mlp_tile_m64_n512_k512": (64, 512, 512),
}

SM110_SHAPES = {
    "pi05_action_gate_up": (51, 16384, 2048),
    "pi05_action_down": (51, 2048, 8192),
    "groot_n17_dit_qkv": (41, 4608, 1536),
    "groot_n17_dit_ffn_up": (41, 6144, 1536),
    "groot_n17_dit_ffn_down": (41, 1536, 6144),
    "groot_legacy_dit_qkv": (51, 4608, 1536),
    "groot_backbone_gate_up": (277, 16384, 2048),
    "cosmos_edge_action": (64, 9216, 2048),
    "lingbot_action_gate_up": (105, 16384, 2048),
}

EPILOGUE_SHAPES = {
    "epilogue_tile": (64, 512, 512),
    "motus_up": (360, 14336, 3072),
    "motus_down": (360, 3072, 14336),
}

MODES = {
    "smoke": ["small_m16_n128_k128"],
    "full": list(SHAPES),
    "thor-models": list(SM110_SHAPES),
}


@dataclass
class Metrics:
    shape: str
    M: int
    N: int
    K: int
    workload: str
    variant: int | None
    max_abs: float
    mean_abs: float
    p99_abs: float
    cosine: float
    passed: bool


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self._ops = getattr(torch.ops, namespace)

    @staticmethod
    def sfa_size_bytes(rows: int, dim: int) -> int:
        n_blocks = dim // 16
        n_row_super = (rows + 127) // 128
        n_col_super = (n_blocks + 3) // 4
        return n_row_super * n_col_super * 512

    def alloc_fp4(self, rows: int, dim: int):
        return (
            torch.empty((rows, dim // 2), device="cuda", dtype=torch.uint8),
            torch.zeros((self.sfa_size_bytes(rows, dim),), device="cuda", dtype=torch.uint8),
        )

    def quantize_fp4_sfa_fp16(self, x, packed, sfa, is_sfb=False):
        self._ops.quantize_fp4_sfa_fp16(x, packed, sfa, bool(is_sfb))

    def quantize_fp4_sfa_mse_fp16(self, x, packed, sfa, is_sfb=False):
        self._ops.quantize_fp4_sfa_mse_fp16(x, packed, sfa, bool(is_sfb))

    def quantize_fp4_sfa_mse_bf16(self, x, packed, sfa, is_sfb=False):
        self._ops.quantize_fp4_sfa_mse_bf16(x, packed, sfa, bool(is_sfb))

    def quantize_e0m3_sfa_fp16(self, x, packed, sfa, is_sfb=False):
        self._ops.quantize_e0m3_sfa_fp16(x, packed, sfa, bool(is_sfb))

    def quantize_fp4_sfa_bf16(self, x, packed, sfa, is_sfb=False):
        self._ops.quantize_fp4_sfa_bf16(x, packed, sfa, bool(is_sfb))

    def dequantize_fp4_sfa_fp16(self, packed, sfa, out, is_sfb=False):
        self._ops.dequantize_fp4_sfa_fp16(packed, sfa, out, bool(is_sfb))

    def nvfp4_gemm_bf16(self, a, b, sfa, sfb, out, alpha=1.0, variant=0):
        self._ops.nvfp4_gemm_bf16(a, b, sfa, sfb, out, float(alpha), int(variant))

    def repack_b_interleaved(self, b, out):
        self._ops.fp4_repack_b_interleaved_sm120(b, out)

    def gemv_interleaved(self, a, b, sfa, sfb, out, alpha=1.0, warps=4, stages=4):
        self._ops.fp4_w4a4_gemv_warpsplit_interleaved_bf16(
            a, b, sfa, sfb, out, float(alpha), int(warps), int(stages)
        )

    def gemm_warpsplit_mrows(self, a, b, sfa, sfb, out, alpha=1.0, warps=2, stages=6):
        self._ops.fp4_w4a4_gemm_warpsplit_mrows_bf16(
            a, b, sfa, sfb, out, float(alpha), int(warps), int(stages)
        )

    def m256(self, a, b, sfa, sfb, out, alpha=1.0):
        size = self._ops.nvfp4_gemm_m256_workspace_size(a, b, sfa, sfb)
        workspace = torch.empty(size, device=a.device, dtype=torch.uint8)
        self._ops.nvfp4_gemm_m256_bf16(
            a, b, sfa, sfb, workspace, out, float(alpha)
        )

    def nvfp4_gemm_fp16(self, a, b, sfa, sfb, out, alpha=1.0, variant=-1):
        self._ops.nvfp4_gemm_fp16(
            a, b, sfa, sfb, out, float(alpha), int(variant)
        )

    def nvfp4_gemm_variant_bf16(self, a, b, sfa, sfb, out, alpha=1.0, variant=7):
        self._ops.nvfp4_gemm_variant_bf16(
            a, b, sfa, sfb, out, float(alpha), int(variant)
        )

    def nvfp4_gemm_nvfp4(self, a, b, sfa, sfb, out_packed, out_sfa):
        self._ops.nvfp4_gemm_nvfp4(
            a, b, sfa, sfb, out_packed, out_sfa
        )

    def nvfp4_gemm_geglu_nvfp4_fp16(
        self, a, b, sfa, sfb, scratch, out_packed, out_sfa, skinny=False
    ):
        self._ops.nvfp4_gemm_geglu_nvfp4_fp16(
            a, b, sfa, sfb, scratch, out_packed, out_sfa, bool(skinny)
        )

    def nvfp4_gemm_bias_gelu_nvfp4_fp16(
        self, a, b, sfa, sfb, bias, out_packed, out_sfa
    ):
        self._ops.nvfp4_gemm_bias_gelu_nvfp4_fp16(
            a, b, sfa, sfb, bias, out_packed, out_sfa
        )

    def nvfp4_gemm_bias_residual_fp16(
        self, a, b, sfa, sfb, bias, residual, out
    ):
        self._ops.nvfp4_gemm_bias_residual_fp16(
            a, b, sfa, sfb, bias, residual, out
        )

    def nvfp4_gemm_bias_bf16(self, a, b, sfa, sfb, bias, out):
        self._ops.nvfp4_gemm_bias_bf16(a, b, sfa, sfb, bias, out)

    def nvfp4_gemm_bias_residual_bf16(
        self, a, b, sfa, sfb, bias, residual, out
    ):
        self._ops.nvfp4_gemm_bias_residual_bf16(
            a, b, sfa, sfb, bias, residual, out
        )

    def nvfp4_gemm_residual_bf16(self, a, b, sfa, sfb, residual, out, alpha=1.0):
        self._ops.nvfp4_gemm_residual_bf16(
            a, b, sfa, sfb, residual, out, float(alpha)
        )

    def nvfp4_gemm_bias_gelu_bf16(self, a, b, sfa, sfb, bias, out, alpha=1.0):
        self._ops.nvfp4_gemm_bias_gelu_bf16(
            a, b, sfa, sfb, bias, out, float(alpha)
        )

    def nvfp4_gemm_bias_gelu_nvfp4(
        self, a, b, sfa, sfb, bias, out_packed, out_sfa, alpha=1.0
    ):
        self._ops.nvfp4_gemm_bias_gelu_nvfp4(
            a, b, sfa, sfb, bias, out_packed, out_sfa, float(alpha)
        )

    def nvfp4_gemm_streamk_bf16(self, a, b, sfa, sfb, out, alpha=1.0):
        self._ops.nvfp4_gemm_streamk_bf16(a, b, sfa, sfb, out, float(alpha))

    def nvfp4_gemm_streamk_bias_bf16(
        self, a, b, sfa, sfb, bias, out, alpha=1.0
    ):
        self._ops.nvfp4_gemm_streamk_bias_bf16(
            a, b, sfa, sfb, bias, out, float(alpha)
        )

    def e0m3_weight_gemm_fp16(
        self, a, b, sfa, sfb, out, alpha=1.0, a_format=1
    ):
        self._ops.e0m3_weight_gemm_fp16(
            a, b, sfa, sfb, out, float(alpha), int(a_format)
        )

    def nvfp4_gemm_relu2_nvfp4(
        self, a, b, sfa, sfb, out_packed, out_sfa
    ):
        self._ops.nvfp4_gemm_relu2_nvfp4(
            a, b, sfa, sfb, out_packed, out_sfa
        )

    def adopt_w4a16_marlin(self, weight, weight_scale, global_scale):
        return _adopt_w4a16_marlin(self._ops, weight, weight_scale, global_scale)

    def w4a16_marlin(self, x, weight, weight_scale, global_scale, workspace, out):
        self._ops.nvfp4_w4a16_marlin_bf16(
            x, weight, weight_scale, global_scale, workspace, out
        )


class InstalledOps:
    """Adapt the public return-value API to the in-place test interface."""

    def __init__(self, module) -> None:
        self._module = module
        self._ops = module.ops

    def sfa_size_bytes(self, rows: int, dim: int) -> int:
        return int(self._module.sfa_size_bytes(rows, dim))

    def alloc_fp4(self, rows: int, dim: int):
        return (
            torch.empty((rows, dim // 2), device="cuda", dtype=torch.uint8),
            torch.zeros(
                (self._module.sfa_size_bytes(rows, dim),),
                device="cuda",
                dtype=torch.uint8,
            ),
        )

    def quantize_fp4_sfa_fp16(self, x, packed, sfa, is_sfb=False):
        self._module.quantize_fp4_sfa_fp16(
            x, packed=packed, sfa=sfa, is_sfb=bool(is_sfb)
        )

    def quantize_fp4_sfa_mse_fp16(self, x, packed, sfa, is_sfb=False):
        self._module.quantize_fp4_sfa_mse_fp16(
            x, packed=packed, sfa=sfa, is_sfb=bool(is_sfb)
        )

    def quantize_fp4_sfa_mse_bf16(self, x, packed, sfa, is_sfb=False):
        self._module.quantize_fp4_sfa_mse_bf16(
            x, packed=packed, sfa=sfa, is_sfb=bool(is_sfb)
        )

    def quantize_e0m3_sfa_fp16(self, x, packed, sfa, is_sfb=False):
        self._module.quantize_e0m3_sfa_fp16(
            x, packed=packed, sfa=sfa, is_sfb=bool(is_sfb)
        )

    def quantize_fp4_sfa_bf16(self, x, packed, sfa, is_sfb=False):
        self._module.quantize_fp4_sfa_bf16(
            x, packed=packed, sfa=sfa, is_sfb=bool(is_sfb)
        )

    def dequantize_fp4_sfa_fp16(self, packed, sfa, out, is_sfb=False):
        self._module.dequantize_fp4_sfa_fp16(
            packed, sfa, out=out, is_sfb=bool(is_sfb)
        )

    def nvfp4_gemm_bf16(self, a, b, sfa, sfb, out, alpha=1.0, variant=0):
        self._module.nvfp4_gemm_bf16(
            a,
            b,
            sfa,
            sfb,
            alpha=float(alpha),
            out=out,
            variant=int(variant),
        )

    def repack_b_interleaved(self, b, out):
        self._module.fp4_repack_b_interleaved_sm120(b, out=out)

    def gemv_interleaved(self, a, b, sfa, sfb, out, alpha=1.0, warps=4, stages=4):
        self._module.fp4_w4a4_gemv_warpsplit_interleaved_bf16(
            a,
            b,
            sfa,
            sfb,
            alpha=float(alpha),
            warps=int(warps),
            stages=int(stages),
            out=out,
        )

    def gemm_warpsplit_mrows(self, a, b, sfa, sfb, out, alpha=1.0, warps=2, stages=6):
        self._module.fp4_w4a4_gemm_warpsplit_mrows_bf16(
            a, b, sfa, sfb, alpha=float(alpha), warps=int(warps),
            stages=int(stages), out=out,
        )

    def m256(self, a, b, sfa, sfb, out, alpha=1.0):
        workspace_size = self._module.nvfp4_gemm_m256_workspace_size(
            a, b, sfa, sfb
        )
        workspace = torch.empty(
            workspace_size, device=a.device, dtype=torch.uint8
        )
        self._module.nvfp4_gemm_m256_bf16(
            a,
            b,
            sfa,
            sfb,
            workspace=workspace,
            alpha=float(alpha),
            out=out,
        )

    def nvfp4_gemm_fp16(self, a, b, sfa, sfb, out, alpha=1.0, variant=-1):
        self._module.nvfp4_gemm_fp16(
            a, b, sfa, sfb, alpha=float(alpha), out=out,
            variant=int(variant),
        )

    def nvfp4_gemm_variant_bf16(self, a, b, sfa, sfb, out, alpha=1.0, variant=7):
        self._module.nvfp4_gemm_variant_bf16(
            a, b, sfa, sfb, alpha=float(alpha), out=out, variant=int(variant)
        )

    def nvfp4_gemm_nvfp4(self, a, b, sfa, sfb, out_packed, out_sfa):
        self._module.nvfp4_gemm_nvfp4(
            a, b, sfa, sfb, out_packed=out_packed, out_sfa=out_sfa
        )

    def nvfp4_gemm_geglu_nvfp4_fp16(
        self, a, b, sfa, sfb, scratch, out_packed, out_sfa, skinny=False
    ):
        self._module.nvfp4_gemm_geglu_nvfp4_fp16(
            a, b, sfa, sfb, skinny=bool(skinny), scratch=scratch,
            out_packed=out_packed, out_sfa=out_sfa,
        )

    def nvfp4_gemm_bias_gelu_nvfp4_fp16(
        self, a, b, sfa, sfb, bias, out_packed, out_sfa
    ):
        self._module.nvfp4_gemm_bias_gelu_nvfp4_fp16(
            a, b, sfa, sfb, bias,
            out_packed=out_packed, out_sfa=out_sfa,
        )

    def nvfp4_gemm_bias_residual_fp16(
        self, a, b, sfa, sfb, bias, residual, out
    ):
        self._module.nvfp4_gemm_bias_residual_fp16(
            a, b, sfa, sfb, bias, residual, out=out
        )

    def nvfp4_gemm_bias_bf16(self, a, b, sfa, sfb, bias, out):
        self._module.nvfp4_gemm_bias_bf16(
            a, b, sfa, sfb, bias, out=out
        )

    def nvfp4_gemm_bias_residual_bf16(
        self, a, b, sfa, sfb, bias, residual, out
    ):
        self._module.nvfp4_gemm_bias_residual_bf16(
            a, b, sfa, sfb, bias, residual, out=out
        )

    def nvfp4_gemm_residual_bf16(self, a, b, sfa, sfb, residual, out, alpha=1.0):
        self._module.nvfp4_gemm_residual_bf16(
            a, b, sfa, sfb, residual, alpha=float(alpha), out=out
        )

    def nvfp4_gemm_bias_gelu_bf16(self, a, b, sfa, sfb, bias, out, alpha=1.0):
        self._module.nvfp4_gemm_bias_gelu_bf16(
            a, b, sfa, sfb, bias, alpha=float(alpha), out=out
        )

    def nvfp4_gemm_bias_gelu_nvfp4(
        self, a, b, sfa, sfb, bias, out_packed, out_sfa, alpha=1.0
    ):
        self._module.nvfp4_gemm_bias_gelu_nvfp4(
            a, b, sfa, sfb, bias, alpha=float(alpha),
            out_packed=out_packed, out_sfa=out_sfa,
        )

    def nvfp4_gemm_streamk_bf16(self, a, b, sfa, sfb, out, alpha=1.0):
        self._module.nvfp4_gemm_streamk_bf16(
            a, b, sfa, sfb, alpha=float(alpha), out=out
        )

    def nvfp4_gemm_streamk_bias_bf16(
        self, a, b, sfa, sfb, bias, out, alpha=1.0
    ):
        self._module.nvfp4_gemm_streamk_bias_bf16(
            a, b, sfa, sfb, bias, alpha=float(alpha), out=out
        )

    def e0m3_weight_gemm_fp16(
        self, a, b, sfa, sfb, out, alpha=1.0, a_format=1
    ):
        self._module.e0m3_weight_gemm_fp16(
            a, b, sfa, sfb, alpha=float(alpha),
            a_format=int(a_format), out=out,
        )

    def nvfp4_gemm_relu2_nvfp4(
        self, a, b, sfa, sfb, out_packed, out_sfa
    ):
        self._module.nvfp4_gemm_relu2_nvfp4(
            a, b, sfa, sfb, out_packed=out_packed, out_sfa=out_sfa
        )

    def adopt_w4a16_marlin(self, weight, weight_scale, global_scale):
        return self._module.adopt_nvfp4_w4a16_marlin(
            weight, weight_scale, global_scale
        )

    def w4a16_marlin(self, x, weight, weight_scale, global_scale, workspace, out):
        self._module.nvfp4_w4a16_marlin_bf16(
            x, weight, weight_scale, global_scale,
            workspace=workspace, out=out,
        )


def _adopt_w4a16_marlin(raw_ops, weight, weight_scale, global_scale):
    """Independent source-test copy of the public bind-time layout contract."""
    n, k = weight.shape[0], weight.shape[1] * 2
    qweight_kn = weight.view(torch.int32).T.contiguous()
    weight_marlin = torch.empty(
        (k // 16, n * 2), device=weight.device, dtype=torch.int32
    )
    raw_ops.nvfp4_w4a16_marlin_repack(qweight_kn, weight_marlin)

    scale_perm = [i + 8 * j for i in range(8) for j in range(8)]
    scales = weight_scale.T.contiguous().to(torch.bfloat16)
    scales = scales.reshape(-1, 64)[:, scale_perm].reshape(-1, n).contiguous()
    scales = scales.to(torch.float16)
    scales = scales.view(-1, 4)[:, [0, 2, 1, 3]].reshape(scales.shape)
    scaled = scales.float() * (2**7)
    nonzero = scaled > 0
    scale_factor = 1.0
    if bool(nonzero.any()):
        max_value = scaled[nonzero].max()
        if bool(max_value < 448 * (2**7)):
            scale_factor = float((448 * (2**7) / max_value).log2().floor().exp2())
    if scale_factor > 1.0:
        scales = (scales.float() * scale_factor).to(torch.float16)
    scales = scales * (2**7)
    scales[scales < 2] = 0
    scales = (scales.view(torch.int16) << 1).view(torch.float8_e4m3fn)
    scales = scales[:, 1::2].contiguous()
    adjusted_global_scale = (
        global_scale.float() * (2.0 ** 119) / scale_factor
    ).contiguous()
    workspace = torch.zeros(
        torch.cuda.get_device_properties(weight.device).multi_processor_count,
        device=weight.device,
        dtype=torch.int32,
    )
    return weight_marlin, scales, adjusted_global_scale, workspace


def _current_arch_list() -> str:
    major, minor = torch.cuda.get_device_capability(0)
    if (major, minor) == (11, 0):
        return "11.0a"
    if major >= 12:
        return "12.0a"
    return f"{major}.{minor}"


def load_source_ops() -> SourceOps:
    from torch.utils.cpp_extension import load

    cutlass_include = Path(os.environ.get("FLASHRT_CUTLASS_INCLUDE", str(DEFAULT_CUTLASS_INCLUDE)))
    if not REGISTRATION_INCLUDE.is_dir():
        raise RuntimeError(f"missing kernel-builder registration include: {REGISTRATION_INCLUDE}")
    if not cutlass_include.is_dir():
        raise RuntimeError(f"missing CUTLASS include path: {cutlass_include}")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _current_arch_list())
    namespace = "fp4_gemm_source_test"
    capability = torch.cuda.get_device_capability(0)
    if capability == (11, 0):
        gemm_sources = [
            str(PACKAGE / "csrc" / "gemm" / "fp4" / "cutlass_nvfp4_w4a16_gemm_sm100.cu"),
            str(PACKAGE / "csrc" / "gemm" / "fp4" / "cutlass_fp4_gemm_bias_bf16_sm100.cu"),
            str(PACKAGE / "csrc" / "gemm" / "fp4" / "cutlass_fp4_gemm.cu"),
            str(PACKAGE / "csrc" / "gemm" / "fp4" / "cutlass_fp4_gemm_variants.cu"),
            str(PACKAGE / "csrc" / "gemm" / "fp4" / "cutlass_fp4_gemm_bf16_variants_sm100.cu"),
            str(PACKAGE / "csrc" / "gemm" / "fp4" / "cutlass_fp4_gemm_fp4out.cu"),
            str(PACKAGE / "csrc" / "gemm" / "fp4" / "cutlass_fp4_gemm_geglu_il_sm100.cu"),
            str(PACKAGE / "csrc" / "gemm" / "fp4" / "cutlass_fp4_gemm_siglip_ffn_sm100.cu"),
            str(PACKAGE / "csrc" / "gemm" / "fp4" / "cutlass_fp4_gemm_e0m3w_sm100.cu"),
            str(PACKAGE / "csrc" / "gemm" / "fp4" / "cosmos3_edge_fp4_gemm_relu2_fp4out.cu"),
            str(PACKAGE / "csrc" / "quantize" / "quantize_fp4_sfa_bf16.cu"),
            str(PACKAGE / "csrc" / "gemm" / "fp4" / "sm110_dispatch.cu"),
        ]
        source_define = "-DFLASHRT_FP4_GEMM_SOURCE_SM110_ONLY"
    else:
        gemm_sources = [
            str(PACKAGE / "csrc" / "gemm" / "fp4" / "cutlass_nvfp4_w4a16_gemm_sm120.cu"),
            str(PACKAGE / "csrc" / "gemm" / "fp4" / "fp4_w4a4_mma_warpsplit_sm120.cu"),
            str(PACKAGE / "csrc" / "gemm" / "fp4" / "fp4_w4a4_mma_warpsplit_ilv_sm120.cu"),
            str(PACKAGE / "csrc" / "gemm" / "fp4" / "fp4_w4a4_mma_warpsplit_mrows_sm120.cu"),
            str(PACKAGE / "csrc" / "gemm" / "fp4" / "cutlass_nvfp4_gemm_m256_sm120.cu"),
            str(PACKAGE / "csrc" / "gemm" / "fp4" / "cutlass_nvfp4_gemm_bias_gelu_bf16out_sm120.cu"),
            str(PACKAGE / "csrc" / "gemm" / "fp4" / "cutlass_nvfp4_gemm_bias_gelu_fp4out_sm120.cu"),
            str(PACKAGE / "csrc" / "gemm" / "fp4" / "cutlass_nvfp4_gemm_dn_streamk_bias_sm120.cu"),
            str(PACKAGE / "csrc" / "marlin" / "marlin_runtime.cu"),
            str(PACKAGE / "csrc" / "marlin" / "gptq_marlin_repack.cu"),
            str(PACKAGE / "csrc" / "marlin" / "sm80_kernel_bfloat16_fe2m1f_bfloat16.cu"),
            str(PACKAGE / "csrc" / "marlin" / "w4a16_marlin_sm120.cu"),
        ]
        source_define = None
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            *gemm_sources,
            str(PACKAGE / "csrc" / "quantize" / "quantize_fp4_sfa.cu"),
            str(PACKAGE / "csrc" / "quantize" / "quantize_e0m3_sfa.cu"),
            str(PACKAGE / "csrc" / "dequantize_fp4_sfa.cu"),
        ],
        extra_include_paths=[
            str(PACKAGE / "csrc"),
            str(cutlass_include),
            str(REGISTRATION_INCLUDE),
        ],
        extra_cflags=[flag for flag in ["-O3", "-DCUDA_KERNEL", source_define] if flag],
        extra_cuda_cflags=[
            "-std=c++17",
            "-O3",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-static-global-template-stub=false",
            "-DCUDA_KERNEL",
            "-DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1",
            "-DFLASHRT_HAVE_COSMOS3_EDGE=1",
            *([source_define] if source_define else []),
        ],
        verbose=False,
    )
    return SourceOps(namespace)


def load_installed_ops(artifact: str | None):
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return InstalledOps(importlib.import_module("fp4_gemm"))
    finally:
        if artifact:
            sys.path.remove(artifact)


def make_inputs(m: int, n: int, k: int, seed: int):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    a = (torch.randn((m, k), device="cuda", generator=gen) * 0.25).to(torch.float16).contiguous()
    b = (torch.randn((n, k), device="cuda", generator=gen) * 0.25).to(torch.float16).contiguous()
    return a, b


def metrics(got: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float, float]:
    diff = (got.float() - expected.float()).abs().flatten()
    return (
        float(diff.max().item()),
        float(diff.mean().item()),
        float(torch.quantile(diff, 0.99).item()),
        float(torch.nn.functional.cosine_similarity(got.float().flatten(), expected.float().flatten(), dim=0).item()),
    )


def check_bf16_threshold(max_abs: float, mean_abs: float, p99_abs: float, cosine: float) -> bool:
    return max_abs <= 0.125 and mean_abs <= 0.005 and p99_abs <= 0.03125 and cosine >= 0.999


def select_sm110_variant(shape: tuple[int, int, int]) -> int:
    _m, n, k = shape
    if n >= 4 * k:
        return 1
    if n == 3 * k:
        return 2
    return 0


def prepare_quantized(ops: SourceOps, m: int, n: int, k: int):
    a_fp16, b_fp16 = make_inputs(m, n, k, seed=7000 + m + n + k)
    a_packed, sfa = ops.alloc_fp4(m, k)
    b_packed, sfb = ops.alloc_fp4(n, k)
    ops.quantize_fp4_sfa_fp16(a_fp16, a_packed, sfa, False)
    ops.quantize_fp4_sfa_fp16(b_fp16, b_packed, sfb, True)
    a_deq = torch.empty_like(a_fp16)
    b_deq = torch.empty_like(b_fp16)
    ops.dequantize_fp4_sfa_fp16(a_packed, sfa, a_deq, False)
    ops.dequantize_fp4_sfa_fp16(b_packed, sfb, b_deq, True)
    torch.cuda.synchronize()
    expected = (a_deq.float() @ b_deq.float().T).to(torch.bfloat16)
    return a_packed, b_packed, sfa, sfb, expected


def prepare_quantized_full(ops: SourceOps, m: int, n: int, k: int):
    a_fp16, b_fp16 = make_inputs(m, n, k, seed=9000 + m + n + k)
    a_packed, sfa = ops.alloc_fp4(m, k)
    b_packed, sfb = ops.alloc_fp4(n, k)
    ops.quantize_fp4_sfa_fp16(a_fp16, a_packed, sfa, False)
    ops.quantize_fp4_sfa_fp16(b_fp16, b_packed, sfb, True)
    a_deq = torch.empty_like(a_fp16)
    b_deq = torch.empty_like(b_fp16)
    ops.dequantize_fp4_sfa_fp16(a_packed, sfa, a_deq, False)
    ops.dequantize_fp4_sfa_fp16(b_packed, sfb, b_deq, True)
    return a_packed, b_packed, sfa, sfb, a_deq, b_deq


def dequantize_e0m3_reference(
    packed: torch.Tensor, sfa: torch.Tensor
) -> torch.Tensor:
    """Decode the package's E0M3 bytes and CUTLASS SFA layout independently."""

    rows, packed_dim = packed.shape
    dim = packed_dim * 2
    blocks = dim // 16
    row = torch.arange(rows, device=packed.device).view(rows, 1)
    block = torch.arange(blocks, device=packed.device).view(1, blocks)
    row_block = row >> 7
    row_inner = row & 127
    col_block = block >> 2
    col_inner = block & 3
    n_col_super = (blocks + 3) // 4
    offsets = (
        (row_block * n_col_super + col_block) * 512
        + (row_inner & 31) * 16
        + ((row_inner >> 5) & 3) * 4
        + col_inner
    )
    scale = sfa[offsets].contiguous().view(torch.float8_e4m3fn).float()

    lo = packed & 0x0F
    hi = packed >> 4
    nibble = torch.stack((lo, hi), dim=-1).reshape(rows, dim).to(torch.int16)
    magnitude = (nibble & 0x7).float()
    signed = torch.where((nibble & 0x8) != 0, -magnitude, magnitude)
    return (signed.view(rows, blocks, 16) * scale.unsqueeze(-1)).reshape(
        rows, dim
    )


def run_sm110_e0m3_cosmos_cases(ops) -> tuple[list[Metrics], dict[str, object]]:
    """Validate the additive E0M3 and Cosmos FP4 epilogue surfaces."""

    if torch.cuda.get_device_capability(0) != (11, 0):
        return [], {"skipped": True, "reason": "requires SM110", "passed": True}

    shape = (64, 512, 512)
    m, n, k = shape
    a_fp16, b_fp16 = make_inputs(m, n, k, seed=20260806)
    a_e0m3, sfa_e0m3 = ops.alloc_fp4(m, k)
    b_e0m3, sfb_e0m3 = ops.alloc_fp4(n, k)
    ops.quantize_e0m3_sfa_fp16(a_fp16, a_e0m3, sfa_e0m3, False)
    ops.quantize_e0m3_sfa_fp16(b_fp16, b_e0m3, sfb_e0m3, True)
    torch.cuda.synchronize()

    # Quantization is deterministic and the sign-magnitude payload has no -0.
    a_repeat, sfa_repeat = ops.alloc_fp4(m, k)
    ops.quantize_e0m3_sfa_fp16(a_fp16, a_repeat, sfa_repeat, False)
    torch.cuda.synchronize()
    nibble_lo = a_e0m3 & 0x0F
    nibble_hi = a_e0m3 >> 4
    no_negative_zero = bool(
        torch.all(nibble_lo != 0x8).item()
        and torch.all(nibble_hi != 0x8).item()
    )
    quant_exact = bool(
        torch.equal(a_e0m3, a_repeat)
        and torch.equal(sfa_e0m3, sfa_repeat)
        and no_negative_zero
    )

    a_deq = dequantize_e0m3_reference(a_e0m3, sfa_e0m3).float()
    b_deq = dequantize_e0m3_reference(b_e0m3, sfb_e0m3).float()
    out = torch.empty((m, n), device="cuda", dtype=torch.float16)
    ops.e0m3_weight_gemm_fp16(
        a_e0m3, b_e0m3, sfa_e0m3, sfb_e0m3, out, a_format=0
    )
    torch.cuda.synchronize()
    expected = (a_deq @ b_deq.T).to(torch.float16)
    rows = [result_row("e0m3_tile", shape, "e0m3_weight_gemm_fp16", out, expected)]

    a_nv, b_nv, sfa_nv, sfb_nv, a_nv_deq, b_nv_deq = prepare_quantized_full(
        ops, m, n, k
    )
    relu2_packed, relu2_sfa = ops.alloc_fp4(m, n)
    # Warm-up initializes the native epilogue's scalar before graph capture.
    ops.nvfp4_gemm_relu2_nvfp4(
        a_nv, b_nv, sfa_nv, sfb_nv, relu2_packed, relu2_sfa
    )
    torch.cuda.synchronize()
    relu2_out = torch.empty((m, n), device="cuda", dtype=torch.float16)
    ops.dequantize_fp4_sfa_fp16(relu2_packed, relu2_sfa, relu2_out, False)
    relu2_ref = torch.relu(a_nv_deq.float() @ b_nv_deq.float().T).square().to(
        torch.float16
    )
    staged_packed, staged_sfa = ops.alloc_fp4(m, n)
    ops.quantize_fp4_sfa_fp16(relu2_ref, staged_packed, staged_sfa, False)
    staged_out = torch.empty_like(relu2_out)
    ops.dequantize_fp4_sfa_fp16(staged_packed, staged_sfa, staged_out, False)
    rows.append(result_row(
        "cosmos_relu2_tile", shape, "nvfp4_gemm_relu2_nvfp4",
        relu2_out, staged_out, fp4_output=True,
    ))

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.nvfp4_gemm_relu2_nvfp4(
            a_nv, b_nv, sfa_nv, sfb_nv, relu2_packed, relu2_sfa
        )
    graph.replay()
    torch.cuda.synchronize()
    packed_first = relu2_packed.clone()
    sfa_first = relu2_sfa.clone()
    graph.replay()
    torch.cuda.synchronize()
    graph_exact = bool(
        torch.equal(packed_first, relu2_packed)
        and torch.equal(sfa_first, relu2_sfa)
    )
    gate = {
        "e0m3_quantize_exact": quant_exact,
        "relu2_graph_exact": graph_exact,
        "passed": quant_exact and graph_exact,
    }
    return rows, gate


def run_case(ops: SourceOps, name: str, shape: tuple[int, int, int]) -> list[Metrics]:
    m, n, k = shape
    a_packed, b_packed, sfa, sfb, expected = prepare_quantized(ops, m, n, k)
    results: list[Metrics] = []
    variants = (-1, 0, 1, 2) if torch.cuda.get_device_capability(0) == (11, 0) else (0, 1, 2)
    for variant in variants:
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        ops.nvfp4_gemm_bf16(a_packed, b_packed, sfa, sfb, out, 1.0, variant)
        torch.cuda.synchronize()
        max_abs, mean_abs, p99_abs, cosine = metrics(out, expected)
        results.append(
            Metrics(
                shape=name,
                M=m,
                N=n,
                K=k,
                workload="nvfp4_gemm_bf16",
                variant=variant,
                max_abs=max_abs,
                mean_abs=mean_abs,
                p99_abs=p99_abs,
                cosine=cosine,
                passed=check_bf16_threshold(max_abs, mean_abs, p99_abs, cosine),
            )
        )

    return results


def run_sm120_qwen38_tier_gate(ops) -> dict[str, object]:
    if torch.cuda.get_device_capability(0) != (12, 0):
        return {"skipped": True, "reason": "requires SM120", "passed": True}

    gemv_shapes = (
        (17408, 5120), (5120, 17408), (12288, 5120),
        (1024, 5120), (5120, 6144), (16384, 5120),
    )
    exact_rows = []
    gen = torch.Generator(device="cuda").manual_seed(20260815)
    for n, k in gemv_shapes:
        a = torch.randint(0, 256, (1, k // 2), device="cuda", dtype=torch.uint8, generator=gen)
        b = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8, generator=gen)
        sfa = torch.full((ops.sfa_size_bytes(1, k),), 0x38, device="cuda", dtype=torch.uint8)
        sfb = torch.full((ops.sfa_size_bytes(n, k),), 0x38, device="cuda", dtype=torch.uint8)
        b_interleaved = torch.empty_like(b)
        ops.repack_b_interleaved(b, b_interleaved)
        host_repack = b.view(n // 8, 8, k // 64, 32).permute(0, 2, 1, 3).contiguous().view_as(b)
        repack_exact = torch.equal(b_interleaved, host_repack)
        for warps in (2, 4, 8):
            if (k // 64) % warps:
                continue
            base = torch.empty((1, n), device="cuda", dtype=torch.bfloat16)
            got = torch.empty_like(base)
            ops._ops.fp4_w4a4_gemv_warpsplit_bf16(
                a, b, sfa, sfb, base, 1.0, warps, 3
            )
            ops.gemv_interleaved(
                a, b_interleaved, sfa, sfb, got, warps=warps, stages=3
            )
            torch.cuda.synchronize()
            exact_rows.append({
                "N": n, "K": k, "warps": warps,
                "repack_exact": repack_exact,
                "output_exact": torch.equal(got, base),
            })

    m, n, k = 512, 1024, 1024
    a, b, sfa, sfb, _ = prepare_quantized(ops, m, n, k)
    base = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    got = torch.empty_like(base)
    ops.nvfp4_gemm_bf16(a, b, sfa, sfb, base, variant=0)
    ops.m256(a, b, sfa, sfb, got)
    torch.cuda.synchronize()
    m256_exact = torch.equal(got, base)

    boundary_rejected = False
    try:
        a511 = torch.empty((511, k // 2), device="cuda", dtype=torch.uint8)
        sfa511 = torch.empty(ops.sfa_size_bytes(511, k), device="cuda", dtype=torch.uint8)
        ops._ops.nvfp4_gemm_m256_workspace_size(a511, b, sfa511, sfb)
    except RuntimeError as error:
        boundary_rejected = "M >= 512" in str(error)

    workspace_size = ops._ops.nvfp4_gemm_m256_workspace_size(a, b, sfa, sfb)
    workspace_size_exact = workspace_size == 0
    fake_workspace_size_exact = True
    if hasattr(ops, "_module"):
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            fake_a = torch.empty(a.shape, device="cuda", dtype=a.dtype)
            fake_b = torch.empty(b.shape, device="cuda", dtype=b.dtype)
            fake_sfa = torch.empty(sfa.shape, device="cuda", dtype=sfa.dtype)
            fake_sfb = torch.empty(sfb.shape, device="cuda", dtype=sfb.dtype)
            fake_workspace_size_exact = (
                ops._module.nvfp4_gemm_m256_workspace_size(
                    fake_a, fake_b, fake_sfa, fake_sfb
                )
                == 0
            )
    workspace = torch.empty(workspace_size, device="cuda", dtype=torch.uint8)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops._ops.nvfp4_gemm_m256_bf16(a, b, sfa, sfb, workspace, got, 1.0)
    graph.replay()
    torch.cuda.synchronize()
    first = got.clone()
    graph.replay()
    torch.cuda.synchronize()
    graph_exact = torch.equal(first, got)

    compiled_m256 = (
        torch.compile(
            lambda a_arg: ops._module.nvfp4_gemm_m256_bf16(
                a_arg, b, sfa, sfb, out=got
            ),
            fullgraph=True,
        )
        if hasattr(ops, "_module")
        else None
    )
    if compiled_m256 is not None:
        compiled_m256(a)
        torch.cuda.synchronize()
        compiled_exact = torch.equal(got, base)
    else:
        compiled_exact = True

    mrows_rows = []
    for shape_index, (n, k) in enumerate(gemv_shapes):
        warps, stages = ((4, 4) if n == 1024 else (2, 6))
        b = torch.randint(
            0, 256, (n, k // 2), device="cuda", dtype=torch.uint8,
            generator=gen,
        )
        sfb = torch.full(
            (ops.sfa_size_bytes(n, k),), 0x38,
            device="cuda", dtype=torch.uint8,
        )
        for m in (2, 4, 7, 8):
            x = (
                torch.randn(
                    (m, k), device="cuda", dtype=torch.float32,
                    generator=gen,
                ) * 0.25
            ).bfloat16().contiguous()
            a, sfa = ops.alloc_fp4(m, k)
            ops.quantize_fp4_sfa_bf16(x, a, sfa)
            got_rows = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
            ops.gemm_warpsplit_mrows(
                a, b, sfa, sfb, got_rows, warps=warps, stages=stages
            )
            bit_diffs = 0
            for row in range(m):
                ar, sr = ops.alloc_fp4(1, k)
                ops.quantize_fp4_sfa_bf16(x[row : row + 1], ar, sr)
                expected_row = torch.empty(
                    (1, n), device="cuda", dtype=torch.bfloat16
                )
                ops._ops.fp4_w4a4_gemv_warpsplit_bf16(
                    ar, b, sr, sfb, expected_row, 1.0, warps, stages
                )
                bit_diffs += int(
                    (got_rows[row].view(torch.uint16)
                     != expected_row[0].view(torch.uint16)).sum().item()
                )
            mrows_rows.append({
                "shape_index": shape_index, "M": m, "N": n, "K": k,
                "warps": warps, "stages": stages,
                "bit_diffs_vs_per_row_gemv": bit_diffs,
                "passed": bit_diffs == 0,
            })

    m, n, k = 8, 4096, 4096
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((n, k), device="cuda", dtype=torch.float16)
    a, sfa = ops.alloc_fp4(m, k)
    b, sfb = ops.alloc_fp4(n, k)
    ops.quantize_fp4_sfa_bf16(x, a, sfa)
    ops.quantize_fp4_sfa_fp16(w, b, sfb, True)
    mrows_out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    ops.gemm_warpsplit_mrows(a, b, sfa, sfb, mrows_out)
    mrows_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(mrows_graph):
        ops.gemm_warpsplit_mrows(a, b, sfa, sfb, mrows_out)
    mrows_graph.replay()
    torch.cuda.synchronize()
    mrows_first = mrows_out.clone()
    mrows_graph.replay()
    torch.cuda.synchronize()
    mrows_graph_exact = torch.equal(mrows_first, mrows_out)

    m17 = torch.empty((17, k // 2), device="cuda", dtype=torch.uint8)
    sfa17 = torch.empty(
        ops.sfa_size_bytes(17, k), device="cuda", dtype=torch.uint8
    )
    m17_rejected = False
    try:
        ops.gemm_warpsplit_mrows(
            m17, b, sfa17, sfb,
            torch.empty((17, n), device="cuda", dtype=torch.bfloat16),
        )
    except RuntimeError as error:
        m17_rejected = "1..16 rows" in str(error)
    passed = (
        len(exact_rows) == 18
        and all(row["repack_exact"] and row["output_exact"] for row in exact_rows)
        and len(mrows_rows) == 24
        and all(row["passed"] for row in mrows_rows)
        and mrows_graph_exact and m17_rejected
        and m256_exact and workspace_size_exact and fake_workspace_size_exact
        and boundary_rejected
        and graph_exact and compiled_exact
    )
    return {
        "gemv_rows": exact_rows,
        "m256_vs_tile128_exact": m256_exact,
        "m256_workspace_size_zero": workspace_size_exact,
        "m256_fake_workspace_size_zero": fake_workspace_size_exact,
        "m511_rejected": boundary_rejected,
        "m256_graph_exact": graph_exact,
        "m256_compile_exact": compiled_exact,
        "mrows_rows": mrows_rows,
        "mrows_graph_exact": mrows_graph_exact,
        "mrows_m17_rejected": m17_rejected,
        "passed": passed,
    }


def result_row(name, shape, workload, got, expected, *, fp4_output=False):
    max_abs, mean_abs, p99_abs, cosine = metrics(got, expected)
    if fp4_output:
        mean_magnitude = float(expected.float().abs().mean().item())
        rms = float(expected.float().square().mean().sqrt().item())
        passed = (
            cosine >= 0.9993
            and mean_abs / max(mean_magnitude, 1e-12) <= 0.01
            and p99_abs / max(rms, 1e-12) <= 0.15
        )
    else:
        passed = (
            cosine >= 0.999
            and mean_abs <= 0.008
            and p99_abs <= 0.0625
        )
    return Metrics(
        shape=name,
        M=shape[0],
        N=shape[1],
        K=shape[2],
        workload=workload,
        variant=None,
        max_abs=max_abs,
        mean_abs=mean_abs,
        p99_abs=p99_abs,
        cosine=cosine,
        passed=passed,
    )


def run_epilogue_case(ops, name: str, shape: tuple[int, int, int]):
    m, n, k = shape
    a, b, sfa, sfb, a_deq, b_deq = prepare_quantized_full(ops, m, n, k)
    matmul = a_deq.float() @ b_deq.float().T
    bias = (torch.randn(n, device="cuda") * 0.02).to(torch.bfloat16)
    residual = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)
    rows = []

    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    ops.nvfp4_gemm_bias_bf16(a, b, sfa, sfb, bias, out)
    expected_bias = (matmul + bias.float().view(1, -1)).to(torch.bfloat16)
    rows.append(result_row(
        name, shape, "nvfp4_gemm_bias_bf16", out, expected_bias
    ))

    ops.nvfp4_gemm_residual_bf16(a, b, sfa, sfb, residual, out)
    expected = (matmul + residual.float()).to(torch.bfloat16)
    rows.append(result_row(name, shape, "nvfp4_gemm_residual_bf16", out, expected))

    ops.nvfp4_gemm_bias_gelu_bf16(a, b, sfa, sfb, bias, out)
    expected_gelu = torch.nn.functional.gelu(
        matmul + bias.float().view(1, -1), approximate="tanh"
    ).to(torch.bfloat16)
    rows.append(result_row(name, shape, "nvfp4_gemm_bias_gelu_bf16", out, expected_gelu))

    out_packed, out_sfa = ops.alloc_fp4(m, n)
    ops.nvfp4_gemm_bias_gelu_nvfp4(
        a, b, sfa, sfb, bias, out_packed, out_sfa
    )
    out_deq = torch.empty((m, n), device="cuda", dtype=torch.float16)
    ops.dequantize_fp4_sfa_fp16(out_packed, out_sfa, out_deq, False)
    staged_packed, staged_sfa = ops.alloc_fp4(m, n)
    ops.quantize_fp4_sfa_fp16(
        expected_gelu.to(torch.float16), staged_packed, staged_sfa, False
    )
    staged_deq = torch.empty_like(out_deq)
    ops.dequantize_fp4_sfa_fp16(
        staged_packed, staged_sfa, staged_deq, False
    )
    rows.append(
        result_row(
            name, shape, "nvfp4_gemm_bias_gelu_nvfp4",
            out_deq, staged_deq, fp4_output=True,
        )
    )

    ops.nvfp4_gemm_streamk_bf16(a, b, sfa, sfb, out)
    expected_linear = matmul.to(torch.bfloat16)
    rows.append(result_row(name, shape, "nvfp4_gemm_streamk_bf16", out, expected_linear))

    ops.nvfp4_gemm_streamk_bias_bf16(a, b, sfa, sfb, bias, out)
    rows.append(result_row(name, shape, "nvfp4_gemm_streamk_bias_bf16", out, expected_bias))
    return rows


def run_sm110_epilogue_case(ops, name: str, shape: tuple[int, int, int]):
    m, n, k = shape
    a, b, sfa, sfb, a_deq, b_deq = prepare_quantized_full(ops, m, n, k)
    matmul = a_deq.float() @ b_deq.float().T
    bias = (torch.randn(n, device="cuda") * 0.02).to(torch.bfloat16)
    residual = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)
    rows = []

    out = torch.empty_like(residual)
    ops.nvfp4_gemm_bias_bf16(a, b, sfa, sfb, bias, out)
    expected_bias = (matmul + bias.float().view(1, -1)).to(torch.bfloat16)
    rows.append(result_row(
        name, shape, "nvfp4_gemm_bias_bf16", out, expected_bias
    ))

    before = residual.clone()
    ops.nvfp4_gemm_bias_residual_bf16(
        a, b, sfa, sfb, bias, residual, residual
    )
    expected_residual = (
        matmul + bias.float().view(1, -1) + before.float()
    ).to(torch.bfloat16)
    rows.append(result_row(
        name, shape, "nvfp4_gemm_bias_residual_bf16",
        residual, expected_residual,
    ))

    out_packed, out_sfa = ops.alloc_fp4(m, n)
    ops.nvfp4_gemm_bias_gelu_nvfp4(
        a, b, sfa, sfb, bias, out_packed, out_sfa
    )
    out_deq = torch.empty((m, n), device="cuda", dtype=torch.float16)
    ops.dequantize_fp4_sfa_fp16(out_packed, out_sfa, out_deq, False)
    expected_gelu = torch.nn.functional.gelu(
        matmul + bias.float().view(1, -1), approximate="tanh"
    ).to(torch.bfloat16)
    staged_packed, staged_sfa = ops.alloc_fp4(m, n)
    ops.quantize_fp4_sfa_bf16(
        expected_gelu, staged_packed, staged_sfa, False
    )
    staged_deq = torch.empty_like(out_deq)
    ops.dequantize_fp4_sfa_fp16(
        staged_packed, staged_sfa, staged_deq, False
    )
    rows.append(result_row(
        name, shape, "nvfp4_gemm_bias_gelu_nvfp4",
        out_deq, staged_deq, fp4_output=True,
    ))
    return rows


def _dequantize_linear_nvfp4(weight, weight_scale, global_scale):
    """Dequantize standard ModelOpt row-major NVFP4 weights for the oracle."""
    low = weight & 0x0F
    high = (weight >> 4) & 0x0F
    codes = torch.stack((low, high), dim=-1).flatten(start_dim=1)
    magnitude = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=weight.device,
        dtype=torch.float32,
    )
    values = magnitude[(codes & 0x07).long()]
    values = torch.where((codes & 0x08).bool(), -values, values)
    n, k = values.shape
    scales = weight_scale.float().reshape(n, k // 16, 1)
    return (values.reshape(n, k // 16, 16) * scales * global_scale.float()).reshape(n, k)


def run_sm120_w4a16_marlin_gate(ops):
    if torch.cuda.get_device_capability(0) != (12, 0):
        return {"skipped": True, "reason": "requires SM120", "passed": True}

    n, k = 128, 512
    rows = []
    for m in (1, 2, 4, 7, 8, 16):
        gen = torch.Generator(device="cuda").manual_seed(20260818 + m)
        weight = torch.randint(
            0, 256, (n, k // 2), device="cuda", dtype=torch.uint8,
            generator=gen,
        )
        weight_scale = (
            torch.rand((n, k // 16), device="cuda", generator=gen) * 0.1 + 0.01
        ).to(torch.float8_e4m3fn)
        global_scale = torch.tensor(
            [0.00390625], device="cuda", dtype=torch.float32
        )
        x = (
            torch.randn((m, k), device="cuda", generator=gen) * 0.25
        ).to(torch.bfloat16)
        weight_marlin, scales_marlin, global_marlin, workspace = (
            ops.adopt_w4a16_marlin(weight, weight_scale, global_scale)
        )
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        ops.w4a16_marlin(
            x, weight_marlin, scales_marlin, global_marlin, workspace, out
        )
        reference_weight = _dequantize_linear_nvfp4(
            weight, weight_scale, global_scale
        )
        expected = (x.float() @ reference_weight.T).to(torch.bfloat16)
        max_abs, mean_abs, p99_abs, cosine = metrics(out, expected)
        rows.append({
            "M": m,
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "p99_abs": p99_abs,
            "cosine": cosine,
            "passed": max_abs <= 0.001 and mean_abs <= 0.0001
            and p99_abs <= 0.001 and cosine >= 0.9999,
        })

    m = 8
    weight = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
    weight_scale = (
        torch.rand((n, k // 16), device="cuda") * 0.1 + 0.01
    ).to(torch.float8_e4m3fn)
    global_scale = torch.tensor([0.00390625], device="cuda", dtype=torch.float32)
    weight_marlin, scales_marlin, global_marlin, workspace = (
        ops.adopt_w4a16_marlin(weight, weight_scale, global_scale)
    )
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    eager = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    replay = torch.empty_like(eager)
    ops.w4a16_marlin(x, weight_marlin, scales_marlin, global_marlin, workspace, eager)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.w4a16_marlin(
            x, weight_marlin, scales_marlin, global_marlin, workspace, replay
        )
    graph.replay()
    torch.cuda.synchronize()
    graph_exact = torch.equal(eager, replay)

    compiled_out = torch.empty_like(eager)

    @torch.compile(fullgraph=True)
    def compiled(inp, packed, scales, global_s, locks, result):
        ops.w4a16_marlin(inp, packed, scales, global_s, locks, result)
        return result

    compiled_result = compiled(
        x, weight_marlin, scales_marlin, global_marlin, workspace, compiled_out
    )
    torch.cuda.synchronize()
    compile_exact = torch.equal(eager, compiled_result)

    boundary_rejected = False
    try:
        invalid_out = torch.empty((17, n), device="cuda", dtype=torch.bfloat16)
        ops.w4a16_marlin(
            torch.empty((17, k), device="cuda", dtype=torch.bfloat16),
            weight_marlin, scales_marlin, global_marlin, workspace, invalid_out,
        )
    except (RuntimeError, ValueError):
        boundary_rejected = True

    passed = (
        all(row["passed"] for row in rows)
        and graph_exact and compile_exact and boundary_rejected
    )
    return {
        "rows": rows,
        "graph_exact": graph_exact,
        "compile_exact": compile_exact,
        "m17_rejected": boundary_rejected,
        "passed": passed,
    }


def run_pi05_sm110_native_cases(ops, full_shapes: bool) -> list[Metrics]:
    """Exercise the native PI0.5 FP16-output and compact FP4 epilogues."""

    if torch.cuda.get_device_capability(0) != (11, 0):
        return []

    rows: list[Metrics] = []
    base_shape = (51, 2048, 1024) if full_shapes else (16, 512, 512)
    m, n, k = base_shape
    a, b, sfa, sfb, a_deq, b_deq = prepare_quantized_full(
        ops, m, n, k
    )
    expected = (a_deq.float() @ b_deq.float().T).to(torch.float16)
    # Sweep every schedule on the bounded smoke shape.  The model-shape gate
    # exercises the production auto-dispatch only; running six large schedule
    # candidates back-to-back is an autotune stress test, not the runtime
    # contract, and can retain substantial CUTLASS workspace state.
    variants = (-1,) if full_shapes else (-1, 0, 1, 2, 7, 10)
    for variant in variants:
        out = torch.empty((m, n), device="cuda", dtype=torch.float16)
        ops.nvfp4_gemm_fp16(a, b, sfa, sfb, out, 1.0, variant)
        torch.cuda.synchronize()
        rows.append(result_row(
            f"pi05_fp16_linear_{m}x{n}x{k}", base_shape,
            f"nvfp4_gemm_fp16_v{variant}", out, expected,
        ))

    geglu_shape = (51, 4096, 1024) if full_shapes else (16, 1024, 512)
    m, n_twice, k = geglu_shape
    a_fp16, _ = make_inputs(m, n_twice, k, seed=20260801)
    gate_weight = (
        torch.randn((n_twice // 2, k), device="cuda") * 0.02
    ).to(torch.float16)
    up_weight = (
        torch.randn((n_twice // 2, k), device="cuda") * 0.02
    ).to(torch.float16)
    interleaved = torch.empty(
        (n_twice, k), device="cuda", dtype=torch.float16
    )
    interleaved[0::2] = gate_weight
    interleaved[1::2] = up_weight
    a, sfa = ops.alloc_fp4(m, k)
    b, sfb = ops.alloc_fp4(n_twice, k)
    ops.quantize_fp4_sfa_fp16(a_fp16, a, sfa, False)
    ops.quantize_fp4_sfa_fp16(interleaved, b, sfb, True)
    a_deq = torch.empty_like(a_fp16)
    b_deq = torch.empty_like(interleaved)
    ops.dequantize_fp4_sfa_fp16(a, sfa, a_deq, False)
    ops.dequantize_fp4_sfa_fp16(b, sfb, b_deq, True)
    gate = a_deq.float() @ b_deq[0::2].float().T
    up = a_deq.float() @ b_deq[1::2].float().T
    geglu_ref = (
        torch.nn.functional.gelu(gate, approximate="tanh") * up
    ).to(torch.float16)
    staged_packed, staged_sfa = ops.alloc_fp4(m, n_twice // 2)
    ops.quantize_fp4_sfa_fp16(
        geglu_ref, staged_packed, staged_sfa, False
    )
    staged = torch.empty_like(geglu_ref)
    ops.dequantize_fp4_sfa_fp16(
        staged_packed, staged_sfa, staged, False
    )
    for skinny in (False, True):
        out_packed, out_sfa = ops.alloc_fp4(m, n_twice // 2)
        scratch = torch.empty(
            (m, n_twice // 2), device="cuda", dtype=torch.uint8
        )
        ops.nvfp4_gemm_geglu_nvfp4_fp16(
            a, b, sfa, sfb, scratch, out_packed, out_sfa, skinny
        )
        out = torch.empty_like(geglu_ref)
        ops.dequantize_fp4_sfa_fp16(
            out_packed, out_sfa, out, False
        )
        rows.append(result_row(
            f"pi05_geglu_{m}x{n_twice}x{k}", geglu_shape,
            f"nvfp4_gemm_geglu_nvfp4_fp16_skinny_{skinny}",
            out, staged, fp4_output=True,
        ))

    # SigLIP's logical FFN width is 4304.  The production NVFP4 path pads
    # that physical GEMM dimension to 4320 for the 32-element TMA contract;
    # the extra weights and bias values are zero and are sliced by the host.
    siglip_shape = (768, 4320, 1152) if full_shapes else (64, 512, 512)
    m, n, k = siglip_shape
    a, b, sfa, sfb, a_deq, b_deq = prepare_quantized_full(
        ops, m, n, k
    )
    matmul = a_deq.float() @ b_deq.float().T
    bias = (torch.randn(n, device="cuda") * 0.02).to(torch.float16)
    out_packed, out_sfa = ops.alloc_fp4(m, n)
    ops.nvfp4_gemm_bias_gelu_nvfp4_fp16(
        a, b, sfa, sfb, bias, out_packed, out_sfa
    )
    torch.cuda.synchronize()
    out = torch.empty((m, n), device="cuda", dtype=torch.float16)
    ops.dequantize_fp4_sfa_fp16(out_packed, out_sfa, out, False)
    gelu_ref = torch.nn.functional.gelu(
        matmul + bias.float().view(1, -1), approximate="tanh"
    ).to(torch.float16)
    staged_packed, staged_sfa = ops.alloc_fp4(m, n)
    ops.quantize_fp4_sfa_fp16(
        gelu_ref, staged_packed, staged_sfa, False
    )
    staged = torch.empty_like(out)
    ops.dequantize_fp4_sfa_fp16(
        staged_packed, staged_sfa, staged, False
    )
    rows.append(result_row(
        f"siglip_up_{m}x{n}x{k}", siglip_shape,
        "nvfp4_gemm_bias_gelu_nvfp4_fp16", out, staged,
        fp4_output=True,
    ))

    # Release the large eager-reference temporaries before constructing the
    # independent down-projection case.  Production uses preallocated buffers;
    # retaining both full reference chains here only stresses the test allocator.
    del matmul, gelu_ref, staged, staged_packed, staged_sfa, out
    del out_packed, out_sfa, a, b, sfa, sfb, a_deq, b_deq
    torch.cuda.empty_cache()

    down_shape = (768, 1152, 4320) if full_shapes else (64, 512, 512)
    m, n, k = down_shape
    a, b, sfa, sfb, a_deq, b_deq = prepare_quantized_full(
        ops, m, n, k
    )
    bias = (torch.randn(n, device="cuda") * 0.02).to(torch.float16)
    residual = torch.randn(
        (m, n), device="cuda", dtype=torch.float16
    )
    residual_before = residual.clone()
    # The production residual epilogue is intentionally in-place (C == D).
    # Exercise that exact contract rather than an unsupported out-of-place
    # layout.
    out = residual
    ops.nvfp4_gemm_bias_residual_fp16(
        a, b, sfa, sfb, bias, residual, out
    )
    torch.cuda.synchronize()
    expected = (
        a_deq.float() @ b_deq.float().T
        + bias.float().view(1, -1)
        + residual_before.float()
    ).to(torch.float16)
    rows.append(result_row(
        f"siglip_down_{m}x{n}x{k}", down_shape,
        "nvfp4_gemm_bias_residual_fp16", out, expected,
    ))
    return rows


def check_installed_compile(ops: InstalledOps) -> dict[str, object]:
    a_packed, b_packed, sfa, sfb, _ = prepare_quantized(ops, 128, 128, 128)

    def call(a, b, scale_a, scale_b):
        return ops._module.nvfp4_gemm_bf16(a, b, scale_a, scale_b)

    eager = call(a_packed, b_packed, sfa, sfb)
    compiled = torch.compile(call, fullgraph=True)
    got = compiled(a_packed, b_packed, sfa, sfb)
    torch.cuda.synchronize()
    max_abs = float((got.float() - eager.float()).abs().max().item())
    passed = bool(
        got.dtype == torch.bfloat16
        and got.shape == eager.shape
        and torch.equal(got, eager)
    )
    return {
        "fullgraph": True,
        "dtype": str(got.dtype),
        "shape": list(got.shape),
        "max_abs": max_abs,
        "exact": bool(torch.equal(got, eager)),
        "passed": passed,
    }


def check_bf16_quantizer(ops) -> dict[str, object]:
    """Require the direct BF16 producer to preserve the established layout."""
    cases = [
        (1, 5120, False),
        (1, 6144, False),
        (1, 17408, False),
        (16, 2048, False),
        (128, 512, False),
        (64, 1024, True),
    ]
    rows = []
    for case_index, (m, k, is_sfb) in enumerate(cases):
        torch.manual_seed(8100 + case_index)
        x = (torch.randn((m, k), device="cuda") * 1.5).to(torch.bfloat16)
        direct_packed, direct_sfa = ops.alloc_fp4(m, k)
        compat_packed, compat_sfa = ops.alloc_fp4(m, k)
        # CUTLASS SFA/SFB buffers contain alignment padding that producers do
        # not write or consume. Zero it so a full-buffer equality check still
        # proves every mapped scale byte lands at the same address.
        direct_sfa.zero_()
        compat_sfa.zero_()
        ops.quantize_fp4_sfa_bf16(x, direct_packed, direct_sfa, is_sfb)
        ops.quantize_fp4_sfa_fp16(
            x.to(torch.float16), compat_packed, compat_sfa, is_sfb
        )
        torch.cuda.synchronize()
        packed_exact = bool(torch.equal(direct_packed, compat_packed))
        sfa_exact = bool(torch.equal(direct_sfa, compat_sfa))
        direct_deq = torch.empty((m, k), device="cuda", dtype=torch.float16)
        compat_deq = torch.empty_like(direct_deq)
        ops.dequantize_fp4_sfa_fp16(
            direct_packed, direct_sfa, direct_deq, is_sfb
        )
        ops.dequantize_fp4_sfa_fp16(
            compat_packed, compat_sfa, compat_deq, is_sfb
        )
        torch.cuda.synchronize()
        dequant_exact = bool(torch.equal(direct_deq, compat_deq))
        rows.append(
            {
                "shape": [m, k],
                "is_sfb": is_sfb,
                "packed_exact": packed_exact,
                "sfa_exact": sfa_exact,
                "dequant_exact": dequant_exact,
                "passed": packed_exact and sfa_exact and dequant_exact,
            }
        )
    return {"rows": rows, "passed": all(row["passed"] for row in rows)}


def run_pi05_thor_batch3_cases(ops, full_shapes: bool) -> list[Metrics]:
    """Validate native BF16 v7/v10, FP4-out, and MSE weight packing."""

    if torch.cuda.get_device_capability(0) != (11, 0):
        return []
    shapes = [
        ("decoder_qkv", 10, 2560, 1024, 10),
        ("decoder_o", 10, 1024, 2048, 10),
        ("decoder_gate_up", 10, 8192, 1024, 10),
        ("decoder_down", 10, 1024, 4096, 10),
    ]
    if full_shapes:
        shapes += [
            ("decoder_qkv_m64", 64, 2560, 1024, 10),
            ("decoder_o_m64", 64, 1024, 2048, 10),
            ("decoder_gate_up_m64", 64, 8192, 1024, 10),
            ("decoder_down_m64", 64, 1024, 4096, 10),
            ("encoder_gate_up_m576", 576, 16384, 2048, 7),
            ("encoder_gate_up_m970", 970, 16384, 2048, 7),
            ("encoder_down_m576", 576, 2048, 16384, 7),
            ("encoder_down_m970", 970, 2048, 16384, 7),
            ("siglip_m512", 512, 4304, 1152, 7),
            ("siglip_m768", 768, 4304, 1152, 7),
        ]

    results: list[Metrics] = []
    for index, (name, m, n, k, variant) in enumerate(shapes):
        a, b = make_inputs(m, n, k, seed=2026080900 + index)
        a_packed, sfa = ops.alloc_fp4(m, k)
        b_packed, sfb = ops.alloc_fp4(n, k)
        ops.quantize_fp4_sfa_fp16(a, a_packed, sfa, False)
        ops.quantize_fp4_sfa_fp16(b, b_packed, sfb, True)
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        ops.nvfp4_gemm_variant_bf16(
            a_packed, b_packed, sfa, sfb, out, variant=variant
        )
        a_deq = torch.empty_like(a)
        b_deq = torch.empty_like(b)
        ops.dequantize_fp4_sfa_fp16(a_packed, sfa, a_deq, False)
        ops.dequantize_fp4_sfa_fp16(b_packed, sfb, b_deq, True)
        expected = (a_deq.float() @ b_deq.float().T).to(torch.bfloat16)
        torch.cuda.synchronize()
        max_abs, mean_abs, p99_abs, cosine = metrics(out, expected)
        passed = cosine >= 0.9995 and mean_abs <= 0.02 * max(
            1.0, float(expected.float().abs().mean())
        )
        results.append(Metrics(
            shape=name, M=m, N=n, K=k, workload="native_bf16_variant",
            variant=variant, max_abs=max_abs, mean_abs=mean_abs,
            p99_abs=p99_abs, cosine=cosine, passed=passed,
        ))
        if index == 0:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                ops.nvfp4_gemm_variant_bf16(
                    a_packed, b_packed, sfa, sfb, out, variant=variant
                )
            graph.replay()
            torch.cuda.synchronize()
            first = out.clone()
            graph.replay()
            torch.cuda.synchronize()
            graph_exact = torch.equal(first, out)
            results.append(Metrics(
                shape=f"{name}_graph", M=m, N=n, K=k,
                workload="native_bf16_variant_graph", variant=variant,
                max_abs=0.0 if graph_exact else float("nan"),
                mean_abs=0.0 if graph_exact else float("nan"),
                p99_abs=0.0 if graph_exact else float("nan"),
                cosine=1.0 if graph_exact else float("nan"),
                passed=graph_exact,
            ))

    m, n, k = 10, 1024, 2048
    a, b = make_inputs(m, n, k, seed=2026080990)
    a_packed, sfa = ops.alloc_fp4(m, k)
    b_packed, sfb = ops.alloc_fp4(n, k)
    ops.quantize_fp4_sfa_fp16(a, a_packed, sfa, False)
    ops.quantize_fp4_sfa_fp16(b, b_packed, sfb, True)
    out_packed, out_sfa = ops.alloc_fp4(m, n)
    ops.nvfp4_gemm_nvfp4(
        a_packed, b_packed, sfa, sfb, out_packed, out_sfa
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.nvfp4_gemm_nvfp4(
            a_packed, b_packed, sfa, sfb, out_packed, out_sfa
        )
    graph.replay()
    torch.cuda.synchronize()
    first_packed = out_packed.clone()
    first_sfa = out_sfa.clone()
    graph.replay()
    torch.cuda.synchronize()
    graph_exact = torch.equal(first_packed, out_packed) and torch.equal(
        first_sfa, out_sfa
    )
    out_deq = torch.empty((m, n), device="cuda", dtype=torch.float16)
    a_deq = torch.empty_like(a)
    b_deq = torch.empty_like(b)
    ops.dequantize_fp4_sfa_fp16(out_packed, out_sfa, out_deq, False)
    ops.dequantize_fp4_sfa_fp16(a_packed, sfa, a_deq, False)
    ops.dequantize_fp4_sfa_fp16(b_packed, sfb, b_deq, True)
    expected = (a_deq.float() @ b_deq.float().T).half()
    torch.cuda.synchronize()
    max_abs, mean_abs, p99_abs, cosine = metrics(out_deq, expected)
    results.append(Metrics(
        shape="decoder_o_fp4out", M=m, N=n, K=k,
        workload="native_fp4out", variant=None, max_abs=max_abs,
        mean_abs=mean_abs, p99_abs=p99_abs, cosine=cosine,
        passed=cosine >= 0.99,
    ))
    results.append(Metrics(
        shape="decoder_o_fp4out_graph", M=m, N=n, K=k,
        workload="native_fp4out_graph", variant=None,
        max_abs=0.0 if graph_exact else float("nan"),
        mean_abs=0.0 if graph_exact else float("nan"),
        p99_abs=0.0 if graph_exact else float("nan"),
        cosine=1.0 if graph_exact else float("nan"), passed=graph_exact,
    ))

    weight_fp16 = make_inputs(32, 32, 512, seed=2026080991)[1]
    for dtype, suffix in (
        (torch.float16, "fp16"),
        (torch.bfloat16, "bf16"),
    ):
        weight = weight_fp16.to(dtype)
        rtn_packed, rtn_sfa = ops.alloc_fp4(weight.shape[0], weight.shape[1])
        mse_packed, mse_sfa = ops.alloc_fp4(weight.shape[0], weight.shape[1])
        if dtype == torch.float16:
            ops.quantize_fp4_sfa_fp16(weight, rtn_packed, rtn_sfa, True)
            ops.quantize_fp4_sfa_mse_fp16(weight, mse_packed, mse_sfa, True)
        else:
            ops.quantize_fp4_sfa_bf16(weight, rtn_packed, rtn_sfa, True)
            ops.quantize_fp4_sfa_mse_bf16(weight, mse_packed, mse_sfa, True)
        rtn = torch.empty_like(weight, dtype=torch.float16)
        mse = torch.empty_like(weight, dtype=torch.float16)
        ops.dequantize_fp4_sfa_fp16(rtn_packed, rtn_sfa, rtn, True)
        ops.dequantize_fp4_sfa_fp16(mse_packed, mse_sfa, mse, True)
        rtn_error = float((rtn.float() - weight.float()).square().mean())
        mse_error = float((mse.float() - weight.float()).square().mean())
        results.append(Metrics(
            shape=f"weight_m32_k512_{suffix}", M=32, N=32, K=512,
            workload=f"mse_weight_pack_{suffix}", variant=None,
            max_abs=mse_error, mean_abs=rtn_error, p99_abs=0.0,
            cosine=float(torch.nn.functional.cosine_similarity(
                mse.float().flatten(), weight.float().flatten(), dim=0)),
            passed=mse_error <= rtn_error + 1e-8,
        ))
    return results


def check_installed_padding_contract(ops: InstalledOps) -> dict[str, object]:
    """Validate the bind-time logical-to-physical NVFP4 helper surface."""
    module = ops._module
    required = (
        "aligned_fp4_dim",
        "pack_nvfp4_weight_bf16",
        "quantize_fp4_sfa_padded_bf16",
        "cutlass_fp4_gemm_geglu_il_hw_v10",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        return {"passed": False, "missing": missing}
    up_weight = torch.randn(
        (4304, 128), device="cuda", dtype=torch.bfloat16
    ).mul_(0.02)
    up_bias = torch.randn((4304,), device="cuda", dtype=torch.bfloat16)
    up_packed, up_sfb, up_bias_padded, up_logical = (
        module.pack_nvfp4_weight_bf16(up_weight, up_bias)
    )
    down_weight = torch.randn(
        (128, 4304), device="cuda", dtype=torch.bfloat16
    ).mul_(0.02)
    down_bias = torch.randn((128,), device="cuda", dtype=torch.bfloat16)
    down_packed, down_sfb, down_bias_padded, down_logical = (
        module.pack_nvfp4_weight_bf16(down_weight, down_bias)
    )
    activation = torch.randn(
        (8, 4304), device="cuda", dtype=torch.bfloat16
    ).mul_(0.1)
    a_packed, sfa, logical_k = module.quantize_fp4_sfa_padded_bf16(
        activation
    )
    residual = torch.randn(
        (8, 128), device="cuda", dtype=torch.bfloat16
    ).mul_(0.1)
    got = module.nvfp4_gemm_bias_residual_bf16(
        a_packed, down_packed, sfa, down_sfb, down_bias_padded, residual
    )
    a_dequant = torch.empty(
        (8, 4320), device="cuda", dtype=torch.float16
    )
    w_dequant = torch.empty(
        (128, 4320), device="cuda", dtype=torch.float16
    )
    ops.dequantize_fp4_sfa_fp16(a_packed, sfa, a_dequant, False)
    ops.dequantize_fp4_sfa_fp16(
        down_packed, down_sfb, w_dequant, True
    )
    expected = (
        a_dequant.float() @ w_dequant.float().T
        + down_bias.float()
        + residual.float()
    ).bfloat16()
    max_abs, mean_abs, p99_abs, cosine = metrics(got, expected)
    passed = (
        module.aligned_fp4_dim(4304) == 4320
        and up_logical == (4304, 128)
        and down_logical == (128, 4304)
        and logical_k == 4304
        and tuple(up_packed.shape) == (4320, 64)
        and tuple(up_bias_padded.shape) == (4320,)
        and torch.count_nonzero(up_bias_padded[4304:]).item() == 0
        and tuple(down_packed.shape) == (128, 2160)
        and tuple(a_packed.shape) == (8, 2160)
        and up_sfb.dtype == torch.uint8
        and sfa.dtype == torch.uint8
        and check_bf16_threshold(max_abs, mean_abs, p99_abs, cosine)
    )
    return {
        "passed": bool(passed),
        "logical_up_weight": list(up_logical),
        "physical_up_weight": [4320, 128],
        "logical_down_weight": list(down_logical),
        "physical_down_weight": [128, 4320],
        "logical_activation_k": logical_k,
        "physical_activation_k": 4320,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "p99_abs": p99_abs,
        "cosine": cosine,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--mode", choices=sorted(MODES), default="smoke")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)

    results: list[Metrics] = []
    selected_shapes = SM110_SHAPES if args.mode == "thor-models" else SHAPES
    for name in MODES[args.mode]:
        results.extend(run_case(ops, name, selected_shapes[name]))
    capability = torch.cuda.get_device_capability(0)
    if args.mode == "full" and capability != (11, 0):
        for name, shape in EPILOGUE_SHAPES.items():
            results.extend(run_epilogue_case(ops, name, shape))
    if capability == (11, 0) and args.mode in {"full", "thor-models"}:
        for name in (
            "groot_n17_dit_qkv",
            "groot_n17_dit_ffn_up",
            "groot_n17_dit_ffn_down",
        ):
            results.extend(run_sm110_epilogue_case(
                ops, name, SM110_SHAPES[name]
            ))
    if capability == (11, 0):
        results.extend(run_pi05_sm110_native_cases(
            ops, full_shapes=args.mode == "thor-models"
        ))
        results.extend(run_pi05_thor_batch3_cases(
            ops, full_shapes=args.mode == "thor-models"
        ))
    sm110_additive_rows, sm110_additive_gate = run_sm110_e0m3_cosmos_cases(ops)
    results.extend(sm110_additive_rows)
    sm120_qwen38_gate = (
        run_sm120_qwen38_tier_gate(ops)
        if args.mode == "full"
        else {"skipped": True, "reason": "full mode only", "passed": True}
    )
    sm120_w4a16_marlin_gate = (
        run_sm120_w4a16_marlin_gate(ops)
        if args.mode == "full"
        else {"skipped": True, "reason": "full mode only", "passed": True}
    )
    compile_check = None
    padding_contract_check = None
    bf16_quantizer_check = check_bf16_quantizer(ops)
    if args.backend == "installed" and args.mode == "full":
        compile_check = check_installed_compile(ops)
    if (args.backend == "installed" and capability == (11, 0)
            and args.mode in {"full", "thor-models"}):
        padding_contract_check = check_installed_padding_contract(ops)
    passed = sum(1 for item in results if item.passed)
    total = len(results)
    if compile_check is not None:
        total += 1
        passed += int(bool(compile_check["passed"]))
    if padding_contract_check is not None:
        total += 1
        passed += int(bool(padding_contract_check["passed"]))
    total += 1
    passed += int(bool(bf16_quantizer_check["passed"]))
    total += 1
    passed += int(bool(sm110_additive_gate["passed"]))
    total += 1
    passed += int(bool(sm120_qwen38_gate["passed"]))
    total += 1
    passed += int(bool(sm120_w4a16_marlin_gate["passed"]))
    payload = {
        "backend": args.backend,
        "mode": args.mode,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "passed": passed,
        "total": total,
        "results": [asdict(item) for item in results],
        "compile_check": compile_check,
        "padding_contract_check": padding_contract_check,
        "bf16_quantizer_check": bf16_quantizer_check,
        "sm110_additive_gate": sm110_additive_gate,
        "sm120_qwen38_gate": sm120_qwen38_gate,
        "sm120_w4a16_marlin_gate": sm120_w4a16_marlin_gate,
    }
    print(json.dumps(payload, indent=2))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
