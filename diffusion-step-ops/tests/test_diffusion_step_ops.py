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

    def pack_tail_bf16(self, tail, flat_dim, out=None):
        if out is None:
            out = torch.empty((flat_dim,), device=tail.device, dtype=tail.dtype)
        self._ops.pack_tail_bf16(tail, int(flat_dim), out)
        return out

    def add_bias_zero_tail_bf16(self, input, bias, valid_cols, out=None):
        if out is None:
            out = torch.empty_like(input)
        self._ops.add_bias_zero_tail_bf16(input, bias, int(valid_cols), out)
        return out

    def extract_tail_f32_to_bf16(self, flat, tail_numel, out=None):
        if out is None:
            out = torch.empty((tail_numel,), device=flat.device, dtype=torch.bfloat16)
        self._ops.extract_tail_f32_to_bf16(flat, int(tail_numel), out)
        return out

    def add_bias_pair_bf16(self, input, bias_a, bias_b):
        out = torch.empty_like(input)
        self._ops.add_bias_pair_bf16(input, bias_a, bias_b, out)
        return out

    def unipc_step_f32_bf16(
        self,
        sample,
        velocity,
        prev_m1,
        prev_m2,
        prev_last_sample,
        sigma,
        corrector_order,
        predictor_order,
        corrector_coefficients,
        predictor_coefficients,
    ):
        outputs = [torch.empty_like(sample) for _ in range(3)]
        self._ops.unipc_step_f32_bf16(
            sample,
            velocity,
            prev_m1,
            prev_m2,
            prev_last_sample,
            float(sigma),
            int(corrector_order),
            int(predictor_order),
            *map(float, corrector_coefficients),
            *map(float, predictor_coefficients),
            *outputs,
        )
        return tuple(outputs)


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


def run_tail_tests(ops) -> int:
    count = 0
    for flat_dim, tail_numel in [(32, 7), (257, 51), (4096, 1024)]:
        tail = torch.randn((tail_numel,), device="cuda", dtype=torch.bfloat16)
        got = ops.pack_tail_bf16(tail, flat_dim)
        ref = torch.zeros((flat_dim,), device="cuda", dtype=torch.bfloat16)
        ref[-tail_numel:] = tail
        assert_close(f"pack_tail {flat_dim=} {tail_numel=}", got, ref, 0.0)

        flat = torch.randn((flat_dim,), device="cuda", dtype=torch.float32)
        got = ops.extract_tail_f32_to_bf16(flat, tail_numel)
        ref = flat[-tail_numel:].to(torch.bfloat16)
        assert_close(f"extract_tail {flat_dim=} {tail_numel=}", got, ref, 0.0)
        count += 2

    for rows, cols, valid_cols in [(1, 16, 7), (51, 64, 32), (105, 257, 256)]:
        input = torch.randn((rows, cols), device="cuda", dtype=torch.bfloat16)
        bias = torch.randn((cols,), device="cuda", dtype=torch.bfloat16)
        got = ops.add_bias_zero_tail_bf16(input, bias, valid_cols)
        ref = (input.float() + bias.float()).to(torch.bfloat16)
        ref[:, valid_cols:] = 0
        assert_close(
            f"add_bias_zero_tail {rows=} {cols=} {valid_cols=}",
            got,
            ref,
            0.0,
        )

        bias_b = torch.randn((cols,), device="cuda", dtype=torch.bfloat16)
        got = ops.add_bias_pair_bf16(input, bias, bias_b)
        ref = (input.float() + bias.float()).to(torch.bfloat16)
        ref = (ref.float() + bias_b.float()).to(torch.bfloat16)
        assert_close(f"add_bias_pair {rows=} {cols=}", got, ref, 0.0)
        count += 2

    tail = torch.randn((51,), device="cuda", dtype=torch.bfloat16)
    input = torch.randn((51, 64), device="cuda", dtype=torch.bfloat16)
    bias_a = torch.randn((64,), device="cuda", dtype=torch.bfloat16)
    bias_b = torch.randn((64,), device="cuda", dtype=torch.bfloat16)

    def invoke(tail, input, bias_a, bias_b):
        return (
            ops.pack_tail_bf16(tail, 257),
            ops.add_bias_pair_bf16(input, bias_a, bias_b),
        )

    eager = invoke(tail, input, bias_a, bias_b)
    compiled = torch.compile(invoke, fullgraph=True)(tail, input, bias_a, bias_b)
    for got, expected in zip(compiled, eager):
        torch.testing.assert_close(got, expected, rtol=0.0, atol=0.0)
    print("PASS action-tail torch.compile fullgraph")
    return count + 1


def run_cosmos_edge_contract(ops) -> int:
    flat_dim = 1_201_920
    tail_numel = 60 * 64
    rows, cols, valid_cols = 60, 64, 9

    tail = torch.randn((tail_numel,), device="cuda", dtype=torch.bfloat16)
    flat = torch.randn((flat_dim,), device="cuda", dtype=torch.float32)
    matrix = torch.randn((rows, cols), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((cols,), device="cuda", dtype=torch.bfloat16)
    packed = torch.empty((flat_dim,), device="cuda", dtype=torch.bfloat16)
    extracted = torch.empty((tail_numel,), device="cuda", dtype=torch.bfloat16)
    biased = torch.empty_like(matrix)

    ops.pack_tail_bf16(tail, flat_dim, out=packed)
    ops.extract_tail_f32_to_bf16(flat, tail_numel, out=extracted)
    ops.add_bias_zero_tail_bf16(matrix, bias, valid_cols, out=biased)
    expected_packed = torch.zeros_like(packed)
    expected_packed[-tail_numel:] = tail
    expected_extracted = flat[-tail_numel:].to(torch.bfloat16)
    expected_biased = (matrix.float() + bias.float()).to(torch.bfloat16)
    expected_biased[:, valid_cols:] = 0
    torch.testing.assert_close(packed, expected_packed, rtol=0.0, atol=0.0)
    torch.testing.assert_close(extracted, expected_extracted, rtol=0.0, atol=0.0)
    torch.testing.assert_close(biased, expected_biased, rtol=0.0, atol=0.0)

    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        ops.pack_tail_bf16(tail, flat_dim, out=packed)
        ops.extract_tail_f32_to_bf16(flat, tail_numel, out=extracted)
        ops.add_bias_zero_tail_bf16(matrix, bias, valid_cols, out=biased)
    graph.replay()
    torch.cuda.synchronize()
    first = (packed.clone(), extracted.clone(), biased.clone())
    graph.replay()
    torch.cuda.synchronize()
    second = (packed.clone(), extracted.clone(), biased.clone())
    for got, expected in zip(second, first):
        torch.testing.assert_close(got, expected, rtol=0.0, atol=0.0)
    print("PASS Cosmos3-Edge action-tail contract and CUDA Graph replay")
    return 4


def run_unipc_tests(ops) -> int:
    count = 0
    corrector = (0.75, 0.2, -0.1, 0.05, 0.4)
    predictor = (0.8, 0.3, -0.07)
    for shape in [(1,), (257,), (1, 16, 17, 8, 8)]:
        sample = torch.randn(shape, device="cuda", dtype=torch.float32)
        velocity = torch.randn(
            shape, device="cuda", dtype=torch.bfloat16
        )
        prev_m1 = torch.randn_like(sample)
        prev_m2 = torch.randn_like(sample)
        prev_last = torch.randn_like(sample)
        for corrector_order, predictor_order in [(0, 1), (1, 1), (1, 2), (2, 2)]:
            got_next, got_m, got_last = ops.unipc_step_f32_bf16(
                sample,
                velocity,
                prev_m1,
                prev_m2,
                prev_last,
                0.37,
                corrector_order,
                predictor_order,
                corrector,
                predictor,
            )
            sigma_velocity = (velocity.float() * 0.37).to(
                torch.bfloat16
            ).float()
            expected_m = sample - sigma_velocity
            expected_last = corrector[0] * sample + corrector[4] * expected_m
            if corrector_order >= 1:
                expected_last = (
                    expected_last
                    + corrector[1] * prev_last
                    + corrector[2] * prev_m1
                )
            if corrector_order >= 2:
                expected_last = expected_last + corrector[3] * prev_m2
            expected_next = (
                predictor[0] * expected_last + predictor[1] * expected_m
            )
            if predictor_order >= 2:
                expected_next = expected_next + predictor[2] * prev_m1
            torch.testing.assert_close(
                got_m, expected_m, rtol=1e-6, atol=1e-6
            )
            torch.testing.assert_close(
                got_last, expected_last, rtol=2e-6, atol=2e-6
            )
            torch.testing.assert_close(
                got_next, expected_next, rtol=2e-6, atol=2e-6
            )
            count += 1

    sample = torch.randn((257,), device="cuda", dtype=torch.float32)
    velocity = torch.randn(
        (257,), device="cuda", dtype=torch.bfloat16
    )
    history = [torch.randn_like(sample) for _ in range(3)]

    def invoke(sample, velocity, prev_m1, prev_m2, prev_last):
        return ops.unipc_step_f32_bf16(
            sample,
            velocity,
            prev_m1,
            prev_m2,
            prev_last,
            0.37,
            2,
            2,
            corrector,
            predictor,
        )

    eager = invoke(sample, velocity, *history)
    compiled = torch.compile(invoke, fullgraph=True)(
        sample, velocity, *history
    )
    for got, expected in zip(compiled, eager):
        torch.testing.assert_close(got, expected, rtol=0.0, atol=0.0)
    print("PASS unipc_step torch.compile fullgraph")
    return count + 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(0)
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    total = (
        run_elementwise_tests(ops)
        + run_video_tests(ops)
        + run_tail_tests(ops)
        + run_cosmos_edge_contract(ops)
        + run_unipc_tests(ops)
    )
    torch.cuda.synchronize()
    print(f"diffusion-step-ops correctness passed: {total} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
