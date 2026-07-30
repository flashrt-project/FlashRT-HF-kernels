#!/usr/bin/env python3
"""Correctness and compile tests for padding-aware FP8 producers."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "padded-fp8-producers"
REGISTRATION_INCLUDE = (
    ROOT.parent / "kernels" / "kernel-builder" / "src" / "pyproject"
    / "templates" / "torch"
)


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self._ops = getattr(torch.ops, namespace)

    def adaptive_rms_norm_quant_fp8_padded_bf16(
        self, x, weight, gamma, beta, scale, eps=1e-6, *, padded_rows=None,
        output=None
    ):
        padded_rows = x.shape[1] if padded_rows is None else padded_rows
        output = (
            torch.empty(
                (x.shape[0], padded_rows, x.shape[2]),
                device=x.device,
                dtype=torch.float8_e4m3fn,
            )
            if output is None
            else output
        )
        self._ops.adaptive_rms_norm_quant_fp8_padded_bf16(
            x, weight, gamma, beta, scale, float(eps), output
        )
        return output

    def residual_add_adaptive_rms_norm_quant_fp8_padded_bf16(
        self, residual, x, weight, gamma, beta, scale, eps=1e-6, *,
        padded_rows=None, residual_out=None, output=None
    ):
        padded_rows = x.shape[1] if padded_rows is None else padded_rows
        residual_out = torch.empty_like(x) if residual_out is None else residual_out
        output = (
            torch.empty(
                (x.shape[0], padded_rows, x.shape[2]),
                device=x.device,
                dtype=torch.float8_e4m3fn,
            )
            if output is None
            else output
        )
        self._ops.residual_add_adaptive_rms_norm_quant_fp8_padded_bf16(
            residual, x, weight, gamma, beta, scale, float(eps),
            residual_out, output
        )
        return residual_out, output

    def swiglu_quant_fp8_padded_bf16(
        self, gate, up, scale, *, padded_rows=None, output=None
    ):
        padded_rows = gate.shape[0] if padded_rows is None else padded_rows
        output = (
            torch.empty(
                (padded_rows, gate.shape[1]),
                device=gate.device,
                dtype=torch.float8_e4m3fn,
            )
            if output is None
            else output
        )
        self._ops.swiglu_quant_fp8_padded_bf16(gate, up, scale, output)
        return output

    def _merged(self, name, gate_up, scale, padded_rows, output):
        padded_rows = gate_up.shape[0] if padded_rows is None else padded_rows
        output = (
            torch.empty(
                (padded_rows, gate_up.shape[1] // 2),
                device=gate_up.device,
                dtype=torch.float8_e4m3fn,
            )
            if output is None
            else output
        )
        getattr(self._ops, name)(gate_up, scale, output)
        return output

    def swiglu_merged_quant_fp8_padded_bf16(
        self, gate_up, scale, *, padded_rows=None, output=None
    ):
        return self._merged(
            "swiglu_merged_quant_fp8_padded_bf16",
            gate_up, scale, padded_rows, output
        )

    def swiglu_merged_quant_fp8_padded_fp16(
        self, gate_up, scale, *, padded_rows=None, output=None
    ):
        return self._merged(
            "swiglu_merged_quant_fp8_padded_fp16",
            gate_up, scale, padded_rows, output
        )


def _arch_list() -> str:
    major, minor = torch.cuda.get_device_capability(0)
    return "12.0a" if major >= 12 else f"{major}.{minor}"


def load_source_ops() -> SourceOps:
    from torch.utils.cpp_extension import load

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _arch_list())
    namespace = "padded_fp8_producers_source_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "padded_fp8_producers.cu"),
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
        return importlib.import_module("padded_fp8_producers")
    finally:
        if artifact:
            sys.path.remove(artifact)


def quantize_ref(value: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.clamp(value.float() / scale.float(), -448.0, 448.0).to(
        torch.float8_e4m3fn
    )


def adaptive_ref(x, weight, gamma, beta, eps):
    inv_rms = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)
    norm = x.float() * inv_rms * weight.float()
    return ((1.0 + gamma[:, None, :].float()) * norm + beta[:, None, :].float()).to(
        torch.bfloat16
    )


def assert_fp8(name, got, ref):
    mismatch = (got.float() != ref.float()).float().mean().item()
    max_abs = (got.float() - ref.float()).abs().max().item()
    cosine = F.cosine_similarity(
        got.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    print(
        f"{name}: mismatch={mismatch:.8f} max_code_abs={max_abs:.8f} "
        f"cosine={cosine:.8f}"
    )
    if mismatch > 0.002 or cosine < 0.9999:
        raise AssertionError(f"{name}: FP8 code mismatch exceeds contract")


def expect_runtime_error(name, fn):
    try:
        fn()
    except RuntimeError:
        print(f"{name}: rejected")
        return
    raise AssertionError(f"{name}: expected RuntimeError")


def run(ops, mode: str) -> int:
    torch.manual_seed(71)
    shapes = (
        [(1, 51, 2048, 64), (2, 105, 1280, 128)]
        if mode == "smoke"
        else [
            (1, 1, 1280, 16),
            (1, 40, 1536, 64),
            (1, 49, 2048, 64),
            (1, 51, 2048, 64),
            (2, 51, 2048, 64),
            (1, 64, 4096, 64),
            (1, 105, 1280, 128),
            (2, 277, 2048, 320),
        ]
    )
    checks = 0
    for batch, rows, dim, padded in shapes:
        x = (torch.randn((batch, rows, dim), device="cuda") * 0.4).bfloat16()
        residual = (torch.randn_like(x.float()) * 0.3).bfloat16()
        weight = (torch.randn((dim,), device="cuda") * 0.1 + 1.0).bfloat16()
        gamma = (torch.randn((batch, dim), device="cuda") * 0.1).bfloat16()
        beta = (torch.randn((batch, dim), device="cuda") * 0.1).bfloat16()
        scale = torch.tensor([0.0125], device="cuda", dtype=torch.float32)

        got = ops.adaptive_rms_norm_quant_fp8_padded_bf16(
            x, weight, gamma, beta, scale, padded_rows=padded
        )
        ref = quantize_ref(adaptive_ref(x, weight, gamma, beta, 1e-6), scale)
        assert_fp8(f"adaptive B{batch} S{rows} D{dim} P{padded}", got[:, :rows], ref)
        torch.testing.assert_close(
            got[:, rows:].float(), torch.zeros_like(got[:, rows:].float())
        )
        checks += 1

        residual_out, got = (
            ops.residual_add_adaptive_rms_norm_quant_fp8_padded_bf16(
                residual, x, weight, gamma, beta, scale, padded_rows=padded
            )
        )
        residual_ref = (residual.float() + x.float()).bfloat16()
        torch.testing.assert_close(residual_out, residual_ref, rtol=0, atol=0)
        ref = quantize_ref(
            adaptive_ref(residual_ref, weight, gamma, beta, 1e-6), scale
        )
        assert_fp8(
            f"residual-adaptive B{batch} S{rows} D{dim} P{padded}",
            got[:, :rows], ref
        )
        torch.testing.assert_close(
            got[:, rows:].float(), torch.zeros_like(got[:, rows:].float())
        )
        checks += 1

    swiglu_shapes = [(1, 1280, 16), (51, 4096, 64), (105, 3424, 128)]
    for rows, dim, padded in swiglu_shapes:
        gate = (torch.randn((rows, dim), device="cuda") * 0.5).bfloat16()
        up = (torch.randn((rows, dim), device="cuda") * 0.5).bfloat16()
        scale = torch.tensor([0.01], device="cuda", dtype=torch.float32)
        product = (F.silu(gate.float()) * up.float()).bfloat16()
        ref = quantize_ref(product, scale)
        got = ops.swiglu_quant_fp8_padded_bf16(
            gate, up, scale, padded_rows=padded
        )
        assert_fp8(f"swiglu S{rows} D{dim} P{padded}", got[:rows], ref)
        torch.testing.assert_close(
            got[rows:].float(), torch.zeros_like(got[rows:].float())
        )
        checks += 1

        merged = torch.cat((gate, up), dim=-1)
        got_merged = ops.swiglu_merged_quant_fp8_padded_bf16(
            merged, scale, padded_rows=padded
        )
        torch.testing.assert_close(got_merged.float(), got.float(), rtol=0, atol=0)
        checks += 1

        merged_fp16 = merged.float().half()
        got_fp16 = ops.swiglu_merged_quant_fp8_padded_fp16(
            merged_fp16, scale, padded_rows=padded
        )
        ref_fp16 = quantize_ref(
            (F.silu(merged_fp16[:, :dim].float())
             * merged_fp16[:, dim:].float()).bfloat16(),
            scale,
        )
        assert_fp8(
            f"swiglu-fp16 S{rows} D{dim} P{padded}", got_fp16[:rows], ref_fp16
        )
        checks += 1

    x = torch.randn((1, 51, 1280), device="cuda", dtype=torch.bfloat16)
    weight = torch.ones((1280,), device="cuda", dtype=torch.bfloat16)
    gamma = torch.zeros((1, 1280), device="cuda", dtype=torch.bfloat16)
    beta = torch.zeros_like(gamma)
    scale = torch.tensor([0.01], device="cuda")

    def compiled_call(x, weight, gamma, beta, scale, output):
        return ops.adaptive_rms_norm_quant_fp8_padded_bf16(
            x, weight, gamma, beta, scale, output=output
        )

    output = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    eager = compiled_call(x, weight, gamma, beta, scale, output)
    compiled_output = torch.empty_like(output)
    compiled = torch.compile(compiled_call, fullgraph=True)(
        x, weight, gamma, beta, scale, compiled_output
    )
    torch.testing.assert_close(compiled.float(), eager.float(), rtol=0, atol=0)
    checks += 1

    expect_runtime_error(
        "padded_rows_lt_rows",
        lambda: ops.adaptive_rms_norm_quant_fp8_padded_bf16(
            x, weight, gamma, beta, scale, padded_rows=50
        ),
    )
    expect_runtime_error(
        "wrong_dtype",
        lambda: ops.adaptive_rms_norm_quant_fp8_padded_bf16(
            x.float(), weight, gamma, beta, scale
        ),
    )
    expect_runtime_error(
        "non_contiguous",
        lambda: ops.adaptive_rms_norm_quant_fp8_padded_bf16(
            x.transpose(1, 2), weight, gamma, beta, scale
        ),
    )
    checks += 3
    print(f"PASS: {checks} checks ({mode}, {type(ops).__name__})")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("source", "installed"), default="source")
    parser.add_argument("--artifact")
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    args = parser.parse_args()
    ops = (
        load_source_ops()
        if args.backend == "source"
        else load_installed_ops(args.artifact)
    )
    run(ops, args.mode)


if __name__ == "__main__":
    main()
