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
    "groot_dit_qkv": (51, 4608, 1536),
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
            torch.empty((self.sfa_size_bytes(rows, dim),), device="cuda", dtype=torch.uint8),
        )

    def quantize_fp4_sfa_fp16(self, x, packed, sfa, is_sfb=False):
        self._ops.quantize_fp4_sfa_fp16(x, packed, sfa, bool(is_sfb))

    def quantize_fp4_sfa_bf16(self, x, packed, sfa, is_sfb=False):
        self._ops.quantize_fp4_sfa_bf16(x, packed, sfa, bool(is_sfb))

    def dequantize_fp4_sfa_fp16(self, packed, sfa, out, is_sfb=False):
        self._ops.dequantize_fp4_sfa_fp16(packed, sfa, out, bool(is_sfb))

    def nvfp4_gemm_bf16(self, a, b, sfa, sfb, out, alpha=1.0, variant=0):
        self._ops.nvfp4_gemm_bf16(a, b, sfa, sfb, out, float(alpha), int(variant))

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


class InstalledOps:
    """Adapt the public return-value API to the in-place test interface."""

    def __init__(self, module) -> None:
        self._module = module

    def sfa_size_bytes(self, rows: int, dim: int) -> int:
        return int(self._module.sfa_size_bytes(rows, dim))

    def alloc_fp4(self, rows: int, dim: int):
        return (
            torch.empty((rows, dim // 2), device="cuda", dtype=torch.uint8),
            torch.empty(
                (self._module.sfa_size_bytes(rows, dim),),
                device="cuda",
                dtype=torch.uint8,
            ),
        )

    def quantize_fp4_sfa_fp16(self, x, packed, sfa, is_sfb=False):
        self._module.quantize_fp4_sfa_fp16(
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
            str(PACKAGE / "csrc" / "gemm" / "fp4" / "sm110_dispatch.cu"),
        ]
        source_define = "-DFLASHRT_FP4_GEMM_SOURCE_SM110_ONLY"
    else:
        gemm_sources = [
            str(PACKAGE / "csrc" / "gemm" / "fp4" / "cutlass_nvfp4_w4a16_gemm_sm120.cu"),
            str(PACKAGE / "csrc" / "gemm" / "fp4" / "fp4_w4a4_mma_warpsplit_sm120.cu"),
            str(PACKAGE / "csrc" / "gemm" / "fp4" / "cutlass_nvfp4_gemm_bias_gelu_bf16out_sm120.cu"),
            str(PACKAGE / "csrc" / "gemm" / "fp4" / "cutlass_nvfp4_gemm_bias_gelu_fp4out_sm120.cu"),
            str(PACKAGE / "csrc" / "gemm" / "fp4" / "cutlass_nvfp4_gemm_dn_streamk_bias_sm120.cu"),
        ]
        source_define = None
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            *gemm_sources,
            str(PACKAGE / "csrc" / "quantize" / "quantize_fp4_sfa.cu"),
            str(PACKAGE / "csrc" / "dequantize_fp4_sfa.cu"),
        ],
        extra_include_paths=[
            str(PACKAGE / "csrc"),
            str(cutlass_include),
            str(REGISTRATION_INCLUDE),
        ],
        extra_cflags=[flag for flag in ["-O3", "-DCUDA_KERNEL", source_define] if flag],
        extra_cuda_cflags=[
            "-O3",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-DCUDA_KERNEL",
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
    expected_bias = (matmul + bias.float().view(1, -1)).to(torch.bfloat16)
    rows.append(result_row(name, shape, "nvfp4_gemm_streamk_bias_bf16", out, expected_bias))
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
    compile_check = None
    bf16_quantizer_check = check_bf16_quantizer(ops)
    if args.backend == "installed" and args.mode == "full":
        compile_check = check_installed_compile(ops)
    passed = sum(1 for item in results if item.passed)
    total = len(results)
    if compile_check is not None:
        total += 1
        passed += int(bool(compile_check["passed"]))
    total += 1
    passed += int(bool(bf16_quantizer_check["passed"]))
    payload = {
        "backend": args.backend,
        "mode": args.mode,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "passed": passed,
        "total": total,
        "results": [asdict(item) for item in results],
        "compile_check": compile_check,
        "bf16_quantizer_check": bf16_quantizer_check,
    }
    print(json.dumps(payload, indent=2))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
