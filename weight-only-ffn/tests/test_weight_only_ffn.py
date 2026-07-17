#!/usr/bin/env python3
"""Strict source and installed-artifact checks for weight-only-ffn."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "weight-only-ffn"
REGISTRATION_INCLUDE = (
    ROOT.parent / "kernels" / "kernel-builder" / "src" / "pyproject" / "templates" / "torch"
)

LINEAR_SHAPES = {
    "m1_decode": (1, 256, 256),
    "m2_decode": (2, 320, 512),
    "m3_decode": (3, 384, 512),
    "m4_decode": (4, 512, 1024),
}

FFN_SHAPES = {
    "m1_llm": (1, 1024, 2816, 1024),
    "m2_llm": (2, 1024, 2816, 1024),
    "m4_vla": (4, 1024, 4096, 1024),
}


@dataclass
class Check:
    precision: str
    op: str
    shape: str
    max_abs: float
    mean_abs: float
    p99_abs: float
    max_rel: float
    p99_rel: float
    cosine: float
    dtype: str
    passed: bool


def _arch_list() -> str:
    major, minor = torch.cuda.get_device_capability(0)
    if major == 12 and minor == 1:
        return "12.1"
    if major >= 12:
        return "12.0a"
    raise RuntimeError("weight-only-ffn source tests require Blackwell SM120/SM121")


def load_source_module():
    from torch.utils.cpp_extension import load

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _arch_list())
    namespace = "weight_only_ffn_source_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "w4_weight_only.cu"),
            str(PACKAGE / "csrc" / "w4a16_gemm_sm120.cu"),
            str(PACKAGE / "csrc" / "w4a16_matvec_sm120.cu"),
            str(PACKAGE / "csrc" / "w8_weight_only.cu"),
            str(PACKAGE / "csrc" / "ffn_epilogues.cu"),
        ],
        extra_include_paths=[str(PACKAGE / "csrc"), str(REGISTRATION_INCLUDE)],
        extra_cflags=["-O3", "-std=c++17", "-DCUDA_KERNEL"],
        extra_cuda_cflags=["-O3", "-std=c++17", "-DCUDA_KERNEL", "--use_fast_math"],
        is_python_module=False,
        verbose=False,
    )
    return RawOps(getattr(torch.ops, namespace))


def load_installed_module(artifact: str | None):
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return PublicOps(importlib.import_module("weight_only_ffn"))
    finally:
        if artifact:
            sys.path.remove(artifact)


def sfb_bytes(rows: int, cols: int) -> int:
    return ((rows + 127) // 128) * (((cols // 16) + 3) // 4) * 512


class RawOps:
    def __init__(self, ops) -> None:
        self.ops = ops

    def quantize(self, weight: torch.Tensor, bits: int):
        if bits == 4:
            packed = torch.empty((weight.shape[0], weight.shape[1] // 2), device="cuda", dtype=torch.uint8)
            scale = torch.empty((sfb_bytes(*weight.shape),), device="cuda", dtype=torch.uint8)
            self.ops.quantize_w4_weight_bf16(weight, packed, scale)
            dequant = torch.empty_like(weight)
            self.ops.dequantize_w4_weight_bf16(packed, scale, dequant)
        else:
            packed = torch.empty_like(weight, dtype=torch.int8)
            scale = torch.empty((weight.shape[0],), device="cuda", dtype=torch.float32)
            self.ops.quantize_w8_weight_bf16(weight, packed, scale)
            dequant = torch.empty_like(weight)
            self.ops.dequantize_w8_weight_bf16(packed, scale, dequant)
        return packed, scale, dequant

    def linear(self, bits, x, weight, scale, out):
        if bits == 4:
            self.ops.w4a16_linear_bf16(x, weight, scale, 1.0, 3, out)
        else:
            self.ops.w8a16_linear_bf16(x, weight, scale, 0, out)

    def gated(self, bits, x, gu_w, gu_s, dn_w, dn_s, gu_b, dn_b, gelu, gu, hidden, out):
        if bits == 4:
            self.ops.w4a16_gated_ffn_bf16(
                x, gu_w, gu_s, dn_w, dn_s, gu_b, dn_b, gelu,
                1.0, 1.0, 3, gu, hidden, out,
            )
        else:
            self.ops.w8a16_gated_ffn_bf16(
                x, gu_w, gu_s, dn_w, dn_s, gu_b, dn_b, gelu,
                2, gu, hidden, out,
            )

    def gelu(self, bits, x, up_w, up_s, dn_w, dn_s, up_b, dn_b, up, hidden, out):
        if bits == 4:
            self.ops.w4a16_gelu_ffn_bf16(
                x, up_w, up_s, dn_w, dn_s, up_b, dn_b,
                1.0, 1.0, 3, up, hidden, out,
            )
        else:
            self.ops.w8a16_gelu_ffn_bf16(
                x, up_w, up_s, dn_w, dn_s, up_b, dn_b,
                2, up, hidden, out,
            )


class PublicOps(RawOps):
    def __init__(self, module) -> None:
        self.module = module

    def quantize(self, weight: torch.Tensor, bits: int):
        if bits == 4:
            packed, scale = self.module.quantize_w4_weight_bf16(weight)
            dequant = self.module.dequantize_w4_weight_bf16(packed, scale, cols=weight.shape[1])
        else:
            packed, scale = self.module.quantize_w8_weight_bf16(weight)
            dequant = self.module.dequantize_w8_weight_bf16(packed, scale)
        return packed, scale, dequant

    def linear(self, bits, x, weight, scale, out):
        if bits == 4:
            self.module.w4a16_linear_bf16(x, weight, scale, variant=3, out=out)
        else:
            self.module.w8a16_linear_bf16(x, weight, scale, out=out)

    def gated(self, bits, x, gu_w, gu_s, dn_w, dn_s, gu_b, dn_b, gelu, gu, hidden, out):
        fn = getattr(self.module, f"w{bits}a16_{'geglu' if gelu else 'swiglu'}_ffn_bf16")
        fn(x, gu_w, gu_s, dn_w, dn_s, gate_up_bias=gu_b, down_bias=dn_b,
           variant=3 if bits == 4 else 2, workspace=(gu, hidden), out=out)

    def gelu(self, bits, x, up_w, up_s, dn_w, dn_s, up_b, dn_b, up, hidden, out):
        fn = getattr(self.module, f"w{bits}a16_gelu_ffn_bf16")
        fn(x, up_w, up_s, dn_w, dn_s, up_bias=up_b, down_bias=dn_b,
           variant=3 if bits == 4 else 2, workspace=(up, hidden), out=out)


def metrics(got: torch.Tensor, ref: torch.Tensor):
    diff = (got.float() - ref.float()).abs().flatten()
    # Absolute error governs values near zero; relative error becomes useful
    # once the BF16 reference magnitude is large enough to make it stable.
    rel = diff / ref.float().abs().flatten().clamp_min(0.125)
    return (
        float(diff.max()), float(diff.mean()), float(torch.quantile(diff, 0.99)),
        float(rel.max()), float(torch.quantile(rel, 0.99)),
        float(F.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0)),
    )


def record(bits: int, op: str, shape: str, got: torch.Tensor, ref: torch.Tensor) -> Check:
    max_abs, mean_abs, p99_abs, max_rel, p99_rel, cosine = metrics(got, ref)
    passed = (
        got.dtype == torch.bfloat16
        and max_abs <= 0.015625
        and mean_abs <= 0.0005
        and p99_abs <= 0.00390625
        and p99_rel <= 0.02
        and cosine >= 0.9999
    )
    return Check(f"W{bits}A16", op, shape, max_abs, mean_abs, p99_abs, max_rel, p99_rel,
                 cosine, str(got.dtype), passed)


def run(backend, mode: str) -> list[Check]:
    checks: list[Check] = []
    torch.manual_seed(20260717)
    linear_shapes = LINEAR_SHAPES if mode == "full" else {"m1_decode": LINEAR_SHAPES["m1_decode"]}
    ffn_shapes = FFN_SHAPES if mode == "full" else {"m1_llm": FFN_SHAPES["m1_llm"]}

    for shape_name, (m, n, k) in linear_shapes.items():
        x = (torch.randn((m, k), device="cuda") * 0.15).bfloat16()
        weight = (torch.randn((n, k), device="cuda") * 0.04).bfloat16()
        for bits in (4, 8):
            packed, scale, dequant = backend.quantize(weight, bits)
            out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
            backend.linear(bits, x, packed, scale, out)
            ref = (x.float() @ dequant.float().T).bfloat16()
            torch.cuda.synchronize()
            checks.append(record(bits, "linear", shape_name, out, ref))

    for shape_name, (m, k, h, n) in ffn_shapes.items():
        x = (torch.randn((m, k), device="cuda") * 0.12).bfloat16()
        for bits in (4, 8):
            for gelu_gate in (False, True):
                gu_weight = (torch.randn((2 * h, k), device="cuda") * 0.025).bfloat16()
                dn_weight = (torch.randn((n, h), device="cuda") * 0.025).bfloat16()
                gu_bias = (torch.randn((2 * h,), device="cuda") * 0.01).bfloat16()
                dn_bias = (torch.randn((n,), device="cuda") * 0.01).bfloat16()
                gu_w, gu_s, gu_deq = backend.quantize(gu_weight, bits)
                dn_w, dn_s, dn_deq = backend.quantize(dn_weight, bits)
                gu = torch.empty((m, 2 * h), device="cuda", dtype=torch.bfloat16)
                hidden = torch.empty((m, h), device="cuda", dtype=torch.bfloat16)
                out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
                backend.gated(bits, x, gu_w, gu_s, dn_w, dn_s, gu_bias, dn_bias,
                              gelu_gate, gu, hidden, out)
                proj = (x.float() @ gu_deq.float().T).bfloat16()
                gate = proj[:, :h].float() + gu_bias[:h].float()
                up = proj[:, h:].float() + gu_bias[h:].float()
                act = F.gelu(gate, approximate="tanh") if gelu_gate else F.silu(gate)
                hidden_ref = (act * up).bfloat16()
                ref = (hidden_ref.float() @ dn_deq.float().T).bfloat16()
                ref = (ref.float() + dn_bias.float()).bfloat16()
                torch.cuda.synchronize()
                checks.append(record(bits, "geglu" if gelu_gate else "swiglu", shape_name, out, ref))

            up_weight = (torch.randn((h, k), device="cuda") * 0.025).bfloat16()
            dn_weight = (torch.randn((n, h), device="cuda") * 0.025).bfloat16()
            up_bias = (torch.randn((h,), device="cuda") * 0.01).bfloat16()
            dn_bias = (torch.randn((n,), device="cuda") * 0.01).bfloat16()
            up_w, up_s, up_deq = backend.quantize(up_weight, bits)
            dn_w, dn_s, dn_deq = backend.quantize(dn_weight, bits)
            up = torch.empty((m, h), device="cuda", dtype=torch.bfloat16)
            hidden = torch.empty_like(up)
            out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
            backend.gelu(bits, x, up_w, up_s, dn_w, dn_s, up_bias, dn_bias,
                         up, hidden, out)
            proj = (x.float() @ up_deq.float().T).bfloat16()
            hidden_ref = F.gelu(proj.float() + up_bias.float(), approximate="tanh").bfloat16()
            ref = (hidden_ref.float() @ dn_deq.float().T).bfloat16()
            ref = (ref.float() + dn_bias.float()).bfloat16()
            torch.cuda.synchronize()
            checks.append(record(bits, "gelu", shape_name, out, ref))

    # Unsupported M must fail loudly in the default production dispatch.
    x = torch.zeros((5, 256), device="cuda", dtype=torch.bfloat16)
    weight = torch.zeros((256, 256), device="cuda", dtype=torch.bfloat16)
    for bits in (4, 8):
        packed, scale, _ = backend.quantize(weight, bits)
        out = torch.empty((5, 256), device="cuda", dtype=torch.bfloat16)
        try:
            if bits == 4 and isinstance(backend, PublicOps):
                backend.module.w4a16_linear_bf16(x, packed, scale, out=out)
            elif bits == 4:
                backend.ops.w4a16_linear_bf16(x, packed, scale, 1.0, 0, out)
            else:
                backend.linear(bits, x, packed, scale, out)
        except RuntimeError as exc:
            if "M in [1,4]" not in str(exc):
                raise
        else:
            raise AssertionError(f"W{bits}A16 M=5 must be rejected by auto dispatch")

    # Performance qualification is part of the public contract. Diagnostic
    # variants remain callable, but auto must reject known weak geometries.
    def call_auto_linear(bits: int, m: int, n: int, k: int, should_pass: bool):
        x = torch.zeros((m, k), device="cuda", dtype=torch.bfloat16)
        weight = torch.zeros((n, k), device="cuda", dtype=torch.bfloat16)
        packed, scale, _ = backend.quantize(weight, bits)
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        try:
            if isinstance(backend, PublicOps):
                getattr(backend.module, f"w{bits}a16_linear_bf16")(
                    x, packed, scale, out=out,
                )
            elif bits == 4:
                backend.ops.w4a16_linear_bf16(x, packed, scale, 1.0, 0, out)
            else:
                backend.ops.w8a16_linear_bf16(x, packed, scale, 0, out)
        except RuntimeError as exc:
            if should_pass or "no qualified fast path" not in str(exc):
                raise
        else:
            if not should_pass:
                raise AssertionError(
                    f"W{bits}A16 auto must reject weak M={m}, N={n}, K={k}"
                )

    call_auto_linear(4, 1, 4096, 256, True)
    call_auto_linear(4, 2, 2048, 4096, True)
    call_auto_linear(4, 3, 11008, 4096, False)
    call_auto_linear(8, 4, 1024, 4096, True)
    call_auto_linear(8, 4, 1024, 8192, False)

    if isinstance(backend, PublicOps) and mode == "full":
        x = (torch.randn((2, 512), device="cuda") * 0.1).bfloat16()
        weight = (torch.randn((512, 512), device="cuda") * 0.03).bfloat16()
        packed, scale, _ = backend.quantize(weight, 8)

        def public_linear(inp):
            return backend.module.w8a16_linear_bf16(inp, packed, scale)

        eager = public_linear(x)
        compiled = torch.compile(public_linear, fullgraph=True)
        got = compiled(x)
        torch.cuda.synchronize()
        torch.testing.assert_close(got, eager, rtol=0.0, atol=0.0)
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--json-out")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    backend = load_source_module() if args.backend == "source" else load_installed_module(args.artifact)
    checks = run(backend, args.mode)
    passed = sum(item.passed for item in checks)
    payload = {
        "backend": args.backend,
        "mode": args.mode,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "passed": passed,
        "total": len(checks),
        "checks": [asdict(item) for item in checks],
    }
    print(json.dumps(payload, indent=2))
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
