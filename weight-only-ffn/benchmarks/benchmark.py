#!/usr/bin/env python3
"""Tile/variant benchmark for the production M<=4 weight-only domain."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


SHAPES = {
    "llm_m1": (1, 4096, 11008, 4096),
    "llm_m2": (2, 4096, 11008, 4096),
    "llm_m3": (3, 4096, 11008, 4096),
    "llm_m4": (4, 4096, 11008, 4096),
    "vla_m1": (1, 1024, 4096, 1024),
    "vla_m2": (2, 1024, 4096, 1024),
    "vla_m4": (4, 1024, 4096, 1024),
    "vision_m1": (1, 1536, 6144, 1536),
    "vision_m2": (2, 1536, 6144, 1536),
    "vision_m4": (4, 1536, 6144, 1536),
}


def bench(fn, warmup: int, iterations: int, repeats: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end) * 1000.0 / iterations))
    return statistics.median(samples)


def sfb_bytes(rows: int, cols: int) -> int:
    return ((rows + 127) // 128) * (((cols // 16) + 3) // 4) * 512


class SourceModule:
    def __init__(self, ops) -> None:
        self.ops = ops

    def quantize_w4_weight_bf16(self, weight):
        n, k = weight.shape
        packed = torch.empty((n, k // 2), device="cuda", dtype=torch.uint8)
        scale = torch.empty((sfb_bytes(n, k),), device="cuda", dtype=torch.uint8)
        self.ops.quantize_w4_weight_bf16(weight, packed, scale)
        return packed, scale

    def dequantize_w4_weight_bf16(self, packed, scale, *, cols):
        out = torch.empty((packed.shape[0], cols), device="cuda", dtype=torch.bfloat16)
        self.ops.dequantize_w4_weight_bf16(packed, scale, out)
        return out

    def quantize_w8_weight_bf16(self, weight):
        packed = torch.empty_like(weight, dtype=torch.int8)
        scale = torch.empty((weight.shape[0],), device="cuda", dtype=torch.float32)
        self.ops.quantize_w8_weight_bf16(weight, packed, scale)
        return packed, scale

    def dequantize_w8_weight_bf16(self, packed, scale):
        out = torch.empty_like(packed, dtype=torch.bfloat16)
        self.ops.dequantize_w8_weight_bf16(packed, scale, out)
        return out

    def _gated(self, bits, x, gu_w, gu_s, dn_w, dn_s, *, gelu,
               gate_up_bias, down_bias, variant, workspace, out):
        gu, hidden = workspace
        if bits == 4:
            self.ops.w4a16_gated_ffn_bf16(
                x, gu_w, gu_s, dn_w, dn_s, gate_up_bias, down_bias,
                gelu, 1.0, 1.0, variant, gu, hidden, out,
            )
        else:
            self.ops.w8a16_gated_ffn_bf16(
                x, gu_w, gu_s, dn_w, dn_s, gate_up_bias, down_bias,
                gelu, variant, gu, hidden, out,
            )
        return out

    def w4a16_swiglu_ffn_bf16(self, *args, **kwargs):
        return self._gated(4, *args, gelu=False, **kwargs)

    def w4a16_geglu_ffn_bf16(self, *args, **kwargs):
        return self._gated(4, *args, gelu=True, **kwargs)

    def w8a16_swiglu_ffn_bf16(self, *args, **kwargs):
        return self._gated(8, *args, gelu=False, **kwargs)

    def w8a16_geglu_ffn_bf16(self, *args, **kwargs):
        return self._gated(8, *args, gelu=True, **kwargs)

    def _gelu(self, bits, x, up_w, up_s, dn_w, dn_s, *, up_bias,
              down_bias, variant, workspace, out):
        up, hidden = workspace
        if bits == 4:
            self.ops.w4a16_gelu_ffn_bf16(
                x, up_w, up_s, dn_w, dn_s, up_bias, down_bias,
                1.0, 1.0, variant, up, hidden, out,
            )
        else:
            self.ops.w8a16_gelu_ffn_bf16(
                x, up_w, up_s, dn_w, dn_s, up_bias, down_bias,
                variant, up, hidden, out,
            )
        return out

    def w4a16_gelu_ffn_bf16(self, *args, **kwargs):
        return self._gelu(4, *args, **kwargs)

    def w8a16_gelu_ffn_bf16(self, *args, **kwargs):
        return self._gelu(8, *args, **kwargs)


def load_source_module():
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parents[1]
    registration = root.parent.parent / "kernels" / "kernel-builder" / "src" / "pyproject" / "templates" / "torch"
    major, minor = torch.cuda.get_device_capability(0)
    if major < 12:
        raise RuntimeError("source benchmark requires SM120/SM121")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1" if minor == 1 else "12.0a")
    namespace = "weight_only_ffn_benchmark_source"
    load(
        name=namespace,
        sources=[str(root / path) for path in [
            "torch-ext/torch_binding.cpp", "csrc/w4_weight_only.cu",
            "csrc/w4a16_gemm_sm120.cu", "csrc/w4a16_matvec_sm120.cu",
            "csrc/w8_weight_only.cu", "csrc/ffn_epilogues.cu",
        ]],
        extra_include_paths=[str(root / "csrc"), str(registration)],
        extra_cflags=["-O3", "-std=c++17", "-DCUDA_KERNEL"],
        extra_cuda_cflags=["-O3", "-std=c++17", "-DCUDA_KERNEL", "--use_fast_math"],
        is_python_module=False,
        verbose=False,
    )
    return SourceModule(getattr(torch.ops, namespace))


def load_module(backend: str, artifact: str | None):
    if backend == "source":
        return load_source_module()
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return importlib.import_module("weight_only_ffn")
    finally:
        if artifact:
            sys.path.remove(artifact)


def quantize(module, bits: int, weight: torch.Tensor):
    if bits == 4:
        packed, scale = module.quantize_w4_weight_bf16(weight)
        dequant = module.dequantize_w4_weight_bf16(packed, scale, cols=weight.shape[1])
    else:
        packed, scale = module.quantize_w8_weight_bf16(weight)
        dequant = module.dequantize_w8_weight_bf16(packed, scale)
    return packed, scale, dequant


def run_case(module, name: str, shape, bits: int, activation: str,
             warmup: int, iterations: int):
    m, k, h, n = shape
    gated = activation in {"swiglu", "geglu"}
    up_rows = 2 * h if gated else h
    generator = torch.Generator(device="cuda").manual_seed(
        91000 + m + k + h + n + bits + len(activation)
    )
    x = (torch.randn((m, k), generator=generator, device="cuda") * 0.1).bfloat16()
    up_weight = (torch.randn((up_rows, k), generator=generator, device="cuda") * 0.02).bfloat16()
    down_weight = (torch.randn((n, h), generator=generator, device="cuda") * 0.02).bfloat16()
    up_bias = (torch.randn((up_rows,), generator=generator, device="cuda") * 0.01).bfloat16()
    down_bias = (torch.randn((n,), generator=generator, device="cuda") * 0.01).bfloat16()
    up_packed, up_scale, up_dequant = quantize(module, bits, up_weight)
    down_packed, down_scale, down_dequant = quantize(module, bits, down_weight)
    first = torch.empty((m, up_rows), device="cuda", dtype=torch.bfloat16)
    hidden = torch.empty((m, h), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    if gated:
        fn_name = f"w{bits}a16_{activation}_ffn_bf16"
        kernel_fn = getattr(module, fn_name)

        def kernel(variant: int):
            return kernel_fn(
                x, up_packed, up_scale, down_packed, down_scale,
                gate_up_bias=up_bias, down_bias=down_bias, variant=variant,
                workspace=(first, hidden), out=out,
            )

        def reference():
            merged = F.linear(x, up_dequant, up_bias)
            gate, up = merged.split(h, dim=-1)
            act = F.silu(gate) if activation == "swiglu" else F.gelu(gate, approximate="tanh")
            return F.linear(act * up, down_dequant, down_bias)
    else:
        fn_name = f"w{bits}a16_gelu_ffn_bf16"
        kernel_fn = getattr(module, fn_name)

        def kernel(variant: int):
            return kernel_fn(
                x, up_packed, up_scale, down_packed, down_scale,
                up_bias=up_bias, down_bias=down_bias, variant=variant,
                workspace=(first, hidden), out=out,
            )

        def reference():
            return F.linear(
                F.gelu(F.linear(x, up_dequant, up_bias), approximate="tanh"),
                down_dequant, down_bias,
            )

    eager_us = bench(reference, warmup, iterations)
    compiled = torch.compile(reference, fullgraph=True, mode="max-autotune-no-cudagraphs")
    compiled()
    torch.cuda.synchronize()
    compiled_us = bench(compiled, warmup, iterations)
    variants = {
        str(variant): bench(
            lambda variant=variant: kernel(variant), warmup, iterations
        )
        for variant in (1, 2, 3)
    }
    auto_error = None
    try:
        auto_us = bench(lambda: kernel(0), warmup, iterations)
        kernel(0)
    except RuntimeError as exc:
        if "not qualified" not in str(exc) and "no qualified fast path" not in str(exc):
            raise
        auto_us = None
        auto_error = str(exc)
        best_diagnostic_variant = min(variants, key=variants.get)
        kernel(int(best_diagnostic_variant))
    ref = reference()
    torch.cuda.synchronize()
    diff = (out.float() - ref.float()).abs().flatten()
    cosine = F.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0)
    best_diagnostic_variant = min(variants, key=variants.get)
    best_diagnostic_us = variants[best_diagnostic_variant]
    if auto_us is not None and auto_us > best_diagnostic_us * 1.05:
        raise AssertionError(
            f"{name} W{bits}A16 {activation}: auto {auto_us:.3f} us is more "
            f"than 5% slower than diagnostic variant {best_diagnostic_variant} "
            f"at {best_diagnostic_us:.3f} us"
        )
    return {
        "shape": name,
        "M": m,
        "K": k,
        "H": h,
        "N": n,
        "precision": f"W{bits}A16",
        "op": activation,
        "eager_us": eager_us,
        "compile_us": compiled_us,
        "variant_us": variants,
        "auto_status": "accepted" if auto_us is not None else "rejected",
        "auto_us": auto_us,
        "auto_error": auto_error,
        "auto_speedup_vs_eager": eager_us / auto_us if auto_us is not None else None,
        "auto_speedup_vs_compile": compiled_us / auto_us if auto_us is not None else None,
        "best_diagnostic_variant": int(best_diagnostic_variant),
        "best_diagnostic_us": best_diagnostic_us,
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "p99_abs": float(torch.quantile(diff, 0.99)),
        "cosine": float(cosine),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--json-out")
    args = parser.parse_args()
    module = load_module(args.backend, args.artifact)
    names = ["llm_m1"] if args.mode == "smoke" else list(SHAPES)
    rows = []
    for name in names:
        for bits in (4, 8):
            for activation in ("swiglu", "geglu", "gelu"):
                rows.append(run_case(module, name, SHAPES[name], bits, activation,
                                     args.warmup, args.iterations))
    payload = {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "rows": rows,
    }
    print(json.dumps(payload, indent=2))
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
