#!/usr/bin/env python3
"""Correctness tests for diffusion-step-ops."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import importlib
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "diffusion-step-ops"
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

    def add_bf16(self, a, b):
        out = torch.empty_like(a)
        self._ops.add_bf16_out(a, b, out)
        return out

    def euler_step_bf16(self, latent, velocity, dt):
        out = torch.empty_like(latent)
        self._ops.euler_step_bf16_out(latent, velocity, float(dt), out)
        return out

    def cfg_combine_into_residual_bf16(self, residual, v_cond, v_uncond, beta):
        self._ops.cfg_combine_into_residual_bf16(residual, v_cond, v_uncond, float(beta))
        return residual

    def cfg_combine_into_residual_fp16(self, residual, v_cond, v_uncond, beta):
        self._ops.cfg_combine_into_residual_fp16(residual, v_cond, v_uncond, float(beta))
        return residual

    def teacher_force_first_frame_bf16(self, video_latent, cond_latent):
        self._ops.teacher_force_first_frame_bf16(video_latent, cond_latent)
        return video_latent

    def motus_decode_postprocess_bf16_to_fp32(self, decoded):
        out = torch.empty(
            (decoded.shape[0], decoded.shape[1], decoded.shape[2] - 1, decoded.shape[3], decoded.shape[4]),
            device=decoded.device,
            dtype=torch.float32,
        )
        self._ops.motus_decode_postprocess_bf16_to_fp32(decoded, out)
        return out

    def cast_bf16_to_fp32(self, src):
        dst = torch.empty_like(src, dtype=torch.float32)
        self._ops.cast_bf16_to_fp32(src, dst)
        return dst


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
    namespace = "diffusion_step_ops_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "diffusion_step_ops.cu"),
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
        return importlib.import_module("diffusion_step_ops")
    finally:
        if artifact:
            sys.path.remove(artifact)


def assert_close(name: str, got: torch.Tensor, ref: torch.Tensor, atol: float) -> None:
    diff = (got.float() - ref.float()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    cos = torch.nn.functional.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0).item()
    if max_err > atol or cos < 0.9999:
        raise AssertionError(f"{name}: max_err={max_err:.8f}, mean_err={mean_err:.8f}, cos={cos:.8f}")


def run_elementwise_tests(ops) -> int:
    count = 0
    for shape in [(1024,), (1025,), (4, 4096), (2, 16, 32, 64)]:
        a = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        got = ops.add_bf16(a, b)
        ref = (a.float() + b.float()).to(torch.bfloat16)
        assert_close(f"add_bf16 shape={shape}", got, ref, 0.0)

        dt = -0.125
        got = ops.euler_step_bf16(a, b, dt)
        ref = (a.float() + b.float() * dt).to(torch.bfloat16)
        assert_close(f"euler_step_bf16 shape={shape}", got, ref, 0.0)

        residual = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        residual_ref = residual.clone()
        beta = 4.5
        got = ops.cfg_combine_into_residual_bf16(residual, a, b, beta)
        ref = (residual_ref.float() + b.float() + beta * (a.float() - b.float())).to(torch.bfloat16)
        assert_close(f"cfg_bf16 shape={shape}", got, ref, 0.0)

        ah = a.to(torch.float16)
        bh = b.to(torch.float16)
        residual_h = residual_ref.to(torch.float16)
        residual_h_ref = residual_h.clone()
        got = ops.cfg_combine_into_residual_fp16(residual_h, ah, bh, beta)
        ref = (residual_h_ref.float() + bh.float() + beta * (ah.float() - bh.float())).to(torch.float16)
        assert_close(f"cfg_fp16 shape={shape}", got, ref, 0.0)

        got = ops.cast_bf16_to_fp32(a)
        ref = a.float()
        assert_close(f"cast_bf16_to_fp32 shape={shape}", got, ref, 0.0)
        count += 5
    return count


def run_video_tests(ops) -> int:
    count = 0
    for shape in [(1, 4, 5, 16, 16), (2, 8, 9, 8, 8), (1, 16, 17, 16, 24)]:
        video = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        cond = torch.randn((shape[0], shape[1], shape[3], shape[4]), device="cuda", dtype=torch.bfloat16)
        ref = video.clone()
        ref[:, :, 0] = cond
        got = ops.teacher_force_first_frame_bf16(video.clone(), cond)
        assert_close(f"teacher_force shape={shape}", got, ref, 0.0)

        decoded = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 3.0
        got = ops.motus_decode_postprocess_bf16_to_fp32(decoded)
        ref = ((decoded[:, :, 1:].float() + 1.0) * 0.5).clamp(0.0, 1.0).contiguous()
        assert_close(f"motus_postprocess shape={shape}", got, ref, 0.0)
        count += 2
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(0)
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    total = run_elementwise_tests(ops) + run_video_tests(ops)
    torch.cuda.synchronize()
    print(f"diffusion-step-ops correctness passed: {total} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
