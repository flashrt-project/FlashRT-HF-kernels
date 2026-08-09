#!/usr/bin/env python3
"""Native/wrapper/eager/compile benchmark for padded FP8 producers."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parent
sys.path.insert(0, str(PACKAGE / "tests"))
from test_padded_fp8_producers import load_source_ops  # noqa: E402


def _apply_mem_cap(max_mem_gb: float = 30.0) -> None:
    if not torch.cuda.is_available() or max_mem_gb <= 0:
        return
    total = torch.cuda.get_device_properties(0).total_memory
    cap = int(max_mem_gb * 1024**3)
    if total <= 0 or cap >= total:
        return
    torch.cuda.set_per_process_memory_fraction(cap / total)


SHAPES = [
    ("decode", 1, 1, 1280, 16),
    ("groot-dit", 1, 40, 1536, 64),
    ("vla", 1, 51, 2048, 64),
    ("vision", 1, 105, 1280, 128),
    ("prefill", 2, 277, 2048, 320),
]


def load_ops(backend: str, artifact: str | None):
    if backend == "source":
        return load_source_ops()
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return importlib.import_module("padded_fp8_producers")
    finally:
        if artifact:
            sys.path.remove(artifact)


def load_native():
    from torch.utils.cpp_extension import load

    major, minor = torch.cuda.get_device_capability()
    os.environ.setdefault(
        "TORCH_CUDA_ARCH_LIST", "12.0a" if major >= 12 else f"{major}.{minor}"
    )
    return load(
        name="padded_fp8_producers_native_bench",
        sources=[
            str(PACKAGE / "benchmarks" / "native_binding.cpp"),
            str(PACKAGE / "csrc" / "padded_fp8_producers.cu"),
        ],
        extra_include_paths=[str(PACKAGE / "csrc")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "--expt-relaxed-constexpr",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        ],
        verbose=False,
    )


def bench(fn, warmup=100, iterations=500) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def eager_adaptive(x, weight, gamma, beta, scale, output):
    batch, rows, _ = x.shape
    norm = x.float() * torch.rsqrt(
        x.float().square().mean(dim=-1, keepdim=True) + 1e-6
    )
    value = (
        (1.0 + gamma[:, None, :].float()) * norm * weight.float()
        + beta[:, None, :].float()
    ).bfloat16()
    output[:, :rows].copy_(
        torch.clamp(value.float() / scale, -448.0, 448.0).to(
            torch.float8_e4m3fn
        )
    )
    output[:, rows:].zero_()
    return output


def eager_swiglu(gate, up, scale, output):
    rows = gate.shape[0]
    value = (F.silu(gate.float()) * up.float()).bfloat16()
    output[:rows].copy_(
        torch.clamp(value.float() / scale, -448.0, 448.0).to(
            torch.float8_e4m3fn
        )
    )
    output[rows:].zero_()
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("source", "installed"), default="source")
    parser.add_argument("--artifact")
    parser.add_argument("--max-mem-gb", type=float, default=30.0)
    args = parser.parse_args()
    _apply_mem_cap(args.max_mem_gb)
    ops = load_ops(args.backend, args.artifact)
    native = load_native()
    print("op,shape,native_us,wrapper_us,eager_us,compile_us,wrapper/native")

    for label, batch, rows, dim, padded in SHAPES:
        x = (torch.randn((batch, rows, dim), device="cuda") * 0.4).bfloat16()
        weight = torch.ones((dim,), device="cuda", dtype=torch.bfloat16)
        gamma = torch.zeros((batch, dim), device="cuda", dtype=torch.bfloat16)
        beta = torch.zeros_like(gamma)
        scale = torch.tensor([0.01], device="cuda", dtype=torch.float32)
        native_out = torch.empty(
            (batch, padded, dim), device="cuda", dtype=torch.float8_e4m3fn
        )
        wrapper_out = torch.empty_like(native_out)
        eager_out = torch.empty_like(native_out)
        compile_out = torch.empty_like(native_out)

        native_fn = lambda: native.adaptive(
            x.data_ptr(), weight.data_ptr(), gamma.data_ptr(), beta.data_ptr(),
            scale.data_ptr(), native_out.data_ptr(), batch, rows, padded, dim,
            1e-6
        )
        wrapper_fn = lambda: ops.adaptive_rms_norm_quant_fp8_padded_bf16(
            x, weight, gamma, beta, scale, output=wrapper_out
        )
        eager_fn = lambda: eager_adaptive(
            x, weight, gamma, beta, scale, eager_out
        )
        compiled_call = torch.compile(eager_adaptive, fullgraph=True)
        compile_fn = lambda: compiled_call(
            x, weight, gamma, beta, scale, compile_out
        )
        native_fn()
        wrapper_fn()
        torch.testing.assert_close(
            native_out.float(), wrapper_out.float(), rtol=0, atol=0
        )
        times = [bench(fn) for fn in (native_fn, wrapper_fn, eager_fn, compile_fn)]
        print(
            f"adaptive,{label}:B{batch}S{rows}D{dim}P{padded},"
            f"{times[0]:.3f},{times[1]:.3f},{times[2]:.3f},{times[3]:.3f},"
            f"{times[1] / times[0]:.3f}"
        )

        gate = x.reshape(batch * rows, dim)
        up = torch.randn_like(gate)
        native_swiglu = torch.empty(
            (batch * padded, dim), device="cuda", dtype=torch.float8_e4m3fn
        )
        wrapper_swiglu = torch.empty_like(native_swiglu)
        eager_swiglu_out = torch.empty_like(native_swiglu)
        compile_swiglu_out = torch.empty_like(native_swiglu)
        native_fn = lambda: native.swiglu(
            gate.data_ptr(), up.data_ptr(), scale.data_ptr(),
            native_swiglu.data_ptr(), batch * rows, batch * padded, dim
        )
        wrapper_fn = lambda: ops.swiglu_quant_fp8_padded_bf16(
            gate, up, scale, output=wrapper_swiglu
        )
        eager_fn = lambda: eager_swiglu(
            gate, up, scale, eager_swiglu_out
        )
        compiled_call = torch.compile(eager_swiglu, fullgraph=True)
        compile_fn = lambda: compiled_call(
            gate, up, scale, compile_swiglu_out
        )
        native_fn()
        wrapper_fn()
        torch.testing.assert_close(
            native_swiglu.float(), wrapper_swiglu.float(), rtol=0, atol=0
        )
        times = [bench(fn) for fn in (native_fn, wrapper_fn, eager_fn, compile_fn)]
        print(
            f"swiglu,{label}:S{batch * rows}D{dim}P{batch * padded},"
            f"{times[0]:.3f},{times[1]:.3f},{times[2]:.3f},{times[3]:.3f},"
            f"{times[1] / times[0]:.3f}"
        )


if __name__ == "__main__":
    main()
