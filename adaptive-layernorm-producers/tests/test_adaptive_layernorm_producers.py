#!/usr/bin/env python3
"""Correctness tests for adaptive-layernorm-producers."""

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
PACKAGE = ROOT / "adaptive-layernorm-producers"
REGISTRATION_INCLUDE = (
    ROOT.parent
    / "kernels"
    / "kernel-builder"
    / "src"
    / "pyproject"
    / "templates"
    / "torch"
)
FP8_MAX = 448.0


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self._ops = getattr(torch.ops, namespace)

    def ada_layer_norm_quant_fp8_bf16(self, x, scale, shift, act_scale, eps=1e-5, out=None):
        out = torch.empty_like(x, dtype=torch.float8_e4m3fn) if out is None else out
        self._ops.ada_layer_norm_quant_fp8_bf16(x, scale, shift, act_scale, float(eps), out)
        return out

    def ada_layer_norm_quant_fp8_modfp8_bf16(
        self, x, scale_fp8, shift_fp8, scale_deq, shift_deq, act_scale, eps=1e-5, out=None
    ):
        out = torch.empty_like(x, dtype=torch.float8_e4m3fn) if out is None else out
        self._ops.ada_layer_norm_quant_fp8_modfp8_bf16(
            x, scale_fp8, shift_fp8, scale_deq, shift_deq, act_scale, float(eps), out
        )
        return out

    def awq_ada_layer_norm_quant_fp8_bf16(self, x, scale, shift, inv_s, act_scale, eps=1e-5, out=None):
        out = torch.empty_like(x, dtype=torch.float8_e4m3fn) if out is None else out
        self._ops.awq_ada_layer_norm_quant_fp8_bf16(x, scale, shift, inv_s, act_scale, float(eps), out)
        return out

    def ada_layer_norm_quant_nvfp4_swizzled_bf16(self, x, scale, shift, eps=1e-5, packed=None, sf_swizzled=None):
        packed, sf_swizzled = _nvfp4_out(x, packed, sf_swizzled)
        self._ops.ada_layer_norm_quant_nvfp4_swizzled_bf16(x, scale, shift, float(eps), packed, sf_swizzled)
        return packed, sf_swizzled

    def ada_layer_norm_quant_nvfp4_swizzled_modfp8_bf16(
        self, x, scale_fp8, shift_fp8, scale_deq, shift_deq, eps=1e-5, packed=None, sf_swizzled=None
    ):
        packed, sf_swizzled = _nvfp4_out(x, packed, sf_swizzled)
        self._ops.ada_layer_norm_quant_nvfp4_swizzled_modfp8_bf16(
            x, scale_fp8, shift_fp8, scale_deq, shift_deq, float(eps), packed, sf_swizzled
        )
        return packed, sf_swizzled

    def layer_norm_no_affine_quant_fp8_static_bf16(self, x, act_scale, eps=1e-5, out=None):
        out = torch.empty_like(x, dtype=torch.float8_e4m3fn) if out is None else out
        self._ops.layer_norm_no_affine_quant_fp8_static_bf16(x, act_scale, float(eps), out)
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
    namespace = "adaptive_layernorm_producers_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "ada_layer_norm_fp8.cu"),
            str(PACKAGE / "csrc" / "dit_layer_norm_fp8.cu"),
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
        return importlib.import_module("adaptive_layernorm_producers")
    finally:
        if artifact:
            sys.path.remove(artifact)


def swizzled_sf_size(rows: int, dim: int) -> int:
    n_blocks = dim // 16
    return ((rows + 127) // 128) * ((n_blocks + 3) // 4) * 128 * 64


def _nvfp4_out(x: torch.Tensor, packed=None, sf_swizzled=None):
    if packed is None:
        packed = torch.empty((x.shape[0], x.shape[1] // 2), device=x.device, dtype=torch.uint8)
    if sf_swizzled is None:
        sf_swizzled = torch.zeros((swizzled_sf_size(x.shape[0], x.shape[1]),), device=x.device, dtype=torch.uint8)
    return packed, sf_swizzled


def make_case(rows: int, dim: int):
    x = (torch.randn((rows, dim), device="cuda", dtype=torch.float32) * 0.75).to(torch.bfloat16).contiguous()
    scale = (torch.randn((dim,), device="cuda", dtype=torch.float32) * 0.08).to(torch.bfloat16).contiguous()
    shift = (torch.randn((dim,), device="cuda", dtype=torch.float32) * 0.08).to(torch.bfloat16).contiguous()
    inv_s = (1.0 + torch.rand((dim,), device="cuda", dtype=torch.float32) * 0.2).to(torch.bfloat16).contiguous()
    act_scale = torch.tensor([0.025], device="cuda", dtype=torch.float32)
    scale_deq = torch.tensor([0.0075], device="cuda", dtype=torch.float32)
    shift_deq = torch.tensor([0.0075], device="cuda", dtype=torch.float32)
    scale_fp8 = torch.clamp(scale.float() / scale_deq, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).contiguous()
    shift_fp8 = torch.clamp(shift.float() / shift_deq, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).contiguous()
    return x, scale, shift, inv_s, act_scale, scale_fp8, shift_fp8, scale_deq, shift_deq


def ref_layer_norm_no_affine(x: torch.Tensor, eps: float) -> torch.Tensor:
    return ref_layer_norm_no_affine_f32(x, eps).to(torch.bfloat16)


def ref_layer_norm_no_affine_f32(x: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    mean = xf.mean(dim=-1, keepdim=True)
    var = ((xf - mean) * (xf - mean)).mean(dim=-1, keepdim=True)
    return (xf - mean) * torch.rsqrt(var + eps)


def ref_adaln(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float) -> torch.Tensor:
    norm = ref_layer_norm_no_affine_f32(x, eps)
    return (norm * (1.0 + scale.float()) + shift.float()).to(torch.bfloat16)


def ref_adaln_float_mod(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float) -> torch.Tensor:
    norm = ref_layer_norm_no_affine_f32(x, eps)
    return (norm * (1.0 + scale.float()) + shift.float()).to(torch.bfloat16)


def quant_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x.float() / scale.float().reshape(()), -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)


def f32_to_fp4_e2m1(x: float) -> int:
    if x == 0.0:
        return 0
    sign = 0x8 if x < 0 else 0
    ax = abs(x)
    if ax < 0.25:
        return sign
    if ax < 0.75:
        return sign | 0x1
    if ax < 1.5:
        return sign | 0x2
    if ax < 3.0:
        return sign | 0x3
    if ax < 5.0:
        return sign | 0x6
    return sign | 0x7


def f32_to_ue4m3_ceil(x: float) -> int:
    if x <= 0.0:
        return 0
    if x < 0.0009765625:
        return 1
    mant, exp = math.frexp(x)
    exp -= 1
    frac = mant * 2.0 - 1.0
    mantissa = math.ceil(frac * 8.0)
    if mantissa >= 8:
        mantissa = 0
        exp += 1
    biased_exp = exp + 7
    if biased_exp <= 0:
        return 1
    if biased_exp >= 15:
        return 0x7F
    return (biased_exp << 3) | mantissa


def ue4m3_to_f32(v: int) -> float:
    if v == 0:
        return 0.0
    exp = ((v >> 3) & 0xF) - 7
    mant = v & 0x7
    return math.ldexp(1.0 + mant / 8.0, exp)


def ref_nvfp4(modulated: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, dim = modulated.shape
    assert dim % 16 == 0
    packed = torch.zeros((rows, dim // 2), dtype=torch.uint8)
    sf = torch.zeros((swizzled_sf_size(rows, dim),), dtype=torch.uint8)
    cpu = modulated.float().cpu()
    n_blocks = dim // 16
    n_col_blocks = (n_blocks + 3) // 4
    for row in range(rows):
        block_scales: list[float] = []
        for block in range(n_blocks):
            amax = float(cpu[row, block * 16 : (block + 1) * 16].abs().max().item())
            ue = f32_to_ue4m3_ceil(amax / 6.0)
            rb = row // 128
            ri = row % 128
            cb = block // 4
            ci = block % 4
            out_idx = (rb * n_col_blocks + cb) * 512 + (ri % 32) * 16 + (ri // 32) * 4 + ci
            sf[out_idx] = ue
            block_scales.append(ue4m3_to_f32(ue))
        for p in range(dim // 2):
            i = p * 2
            s0 = block_scales[i // 16]
            s1 = block_scales[(i + 1) // 16]
            lo = f32_to_fp4_e2m1(float(cpu[row, i].item()) / s0 if s0 > 0 else 0.0)
            hi = f32_to_fp4_e2m1(float(cpu[row, i + 1].item()) / s1 if s1 > 0 else 0.0)
            packed[row, p] = (hi << 4) | (lo & 0x0F)
    return packed, sf


def assert_exact(name: str, got: torch.Tensor, expected: torch.Tensor) -> None:
    got_cpu = got.detach().cpu()
    exp_cpu = expected.detach().cpu()
    if not torch.equal(got_cpu, exp_cpu):
        diff = (got_cpu.to(torch.int16) - exp_cpu.to(torch.int16)).abs()
        raise AssertionError(f"{name} mismatch: nonzero={int((diff != 0).sum())} max={int(diff.max())}")
    print(f"PASS {name}: exact")


def assert_fp8_contract(name: str, got: torch.Tensor, expected: torch.Tensor) -> None:
    diff = (got.float() - expected.float()).abs()
    nonzero = int((diff != 0).sum().item())
    max_abs = float(diff.max().item())
    sorted_diff = diff.flatten().sort().values
    p99_abs = float(sorted_diff[min(sorted_diff.numel() - 1, math.ceil(0.99 * sorted_diff.numel()) - 1)].item())
    cosine = float(torch.nn.functional.cosine_similarity(got.float().flatten(), expected.float().flatten(), dim=0).item())
    max_nonzero = max(8, diff.numel() // 100000)
    if p99_abs != 0.0 or nonzero > max_nonzero:
        raise AssertionError(
            f"{name} mismatch: nonzero={nonzero}/{diff.numel()} max_abs={max_abs} "
            f"p99_abs={p99_abs} cosine={cosine}"
        )
    status = "exact" if nonzero == 0 else "fp8-boundary"
    print(
        f"PASS {name}: {status} nonzero={nonzero}/{diff.numel()} "
        f"max_abs={max_abs:.6f} p99_abs={p99_abs:.6f} cosine={cosine:.8f}"
    )


def run_shape(ops, label: str, rows: int, dim: int, eps: float) -> None:
    x, scale, shift, inv_s, act_scale, scale_fp8, shift_fp8, scale_deq, shift_deq = make_case(rows, dim)

    mod = ref_adaln(x, scale, shift, eps)
    got_fp8 = ops.ada_layer_norm_quant_fp8_bf16(x, scale, shift, act_scale, eps)
    assert_fp8_contract(f"{label}/ada_fp8", got_fp8, quant_fp8(mod, act_scale))

    scale_mod = scale_fp8.float() * scale_deq
    shift_mod = shift_fp8.float() * shift_deq
    mod_fp8 = ref_adaln_float_mod(x, scale_mod, shift_mod, eps)
    got_modfp8 = ops.ada_layer_norm_quant_fp8_modfp8_bf16(
        x, scale_fp8, shift_fp8, scale_deq, shift_deq, act_scale, eps
    )
    assert_fp8_contract(f"{label}/ada_modfp8_fp8", got_modfp8, quant_fp8(mod_fp8, act_scale))

    awq_ref = mod.float() * inv_s.float()
    got_awq = ops.awq_ada_layer_norm_quant_fp8_bf16(x, scale, shift, inv_s, act_scale, eps)
    assert_fp8_contract(f"{label}/awq_ada_fp8", got_awq, quant_fp8(awq_ref, act_scale))

    got_noaffine = ops.layer_norm_no_affine_quant_fp8_static_bf16(x, act_scale, eps)
    assert_fp8_contract(f"{label}/no_affine_fp8", got_noaffine, quant_fp8(ref_layer_norm_no_affine(x, eps), act_scale))

    if dim % 16 == 0 and rows <= 260:
        packed = torch.empty((rows, dim // 2), device=x.device, dtype=torch.uint8)
        sf = torch.zeros((swizzled_sf_size(rows, dim),), device=x.device, dtype=torch.uint8)
        got_packed, got_sf = ops.ada_layer_norm_quant_nvfp4_swizzled_bf16(
            x, scale, shift, eps, packed=packed, sf_swizzled=sf
        )
        exp_packed, exp_sf = ref_nvfp4(mod)
        assert_exact(f"{label}/nvfp4_packed", got_packed, exp_packed)
        assert_exact(f"{label}/nvfp4_sf", got_sf, exp_sf)

        packed.zero_()
        sf.zero_()
        got_packed2, got_sf2 = ops.ada_layer_norm_quant_nvfp4_swizzled_modfp8_bf16(
            x, scale_fp8, shift_fp8, scale_deq, shift_deq, eps, packed=packed, sf_swizzled=sf
        )
        exp_packed2, exp_sf2 = ref_nvfp4(mod_fp8)
        assert_exact(f"{label}/nvfp4_modfp8_packed", got_packed2, exp_packed2)
        assert_exact(f"{label}/nvfp4_modfp8_sf", got_sf2, exp_sf2)


def run(args) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(2026)
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    shapes = {
        "decode_action": (16, 2048),
        "wan_video_short": (64, 3072),
        "wan_video_ctx": (256, 3072),
        "wan_video_2k": (2520, 3072),
        "wan_video_4k": (4096, 3072),
    }
    if args.mode == "smoke":
        shapes = {"wan_video_short": shapes["wan_video_short"]}
    for label, (rows, dim) in shapes.items():
        run_shape(ops, label, rows, dim, args.eps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--eps", type=float, default=1e-5)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
