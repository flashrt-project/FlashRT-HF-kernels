#!/usr/bin/env python3
"""Correctness tests for flashrt-fp8-swiglu-ffn."""

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
PACKAGE = ROOT / "flashrt-fp8-swiglu-ffn"
REGISTRATION_INCLUDE = (
    ROOT.parent
    / "kernels"
    / "kernel-builder"
    / "src"
    / "pyproject"
    / "templates"
    / "torch"
)


def fp8_dtype() -> torch.dtype:
    if torch.version.hip is not None and hasattr(torch, "float8_e4m3fnuz"):
        return torch.float8_e4m3fnuz
    return torch.float8_e4m3fn


def fp8_max() -> float:
    return 240.0 if torch.version.hip is not None else 448.0


def midm_padded_rows(rows: int) -> int:
    if (
        torch.version.hip is None
        and torch.cuda.get_device_capability(0) == (11, 0)
        and 9 <= rows <= 128
    ):
        return ((rows + 63) // 64) * 64
    return rows


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self._ops = getattr(torch.ops, namespace)

    def fp8_gemm_bf16(self, x, w, x_scale, w_scale, out=None):
        if out is None:
            out = torch.empty((x.shape[0], w.shape[0]), device=x.device, dtype=torch.bfloat16)
        self._ops.fp8_gemm_bf16(x, w, x_scale, w_scale, out)
        return out

    def silu_mul_merged_quantize_fp8_static_bf16(self, gate_up, scale, out=None):
        if out is None:
            out = torch.empty(
                (gate_up.shape[0], gate_up.shape[1] // 2),
                device=gate_up.device,
                dtype=fp8_dtype(),
            )
        self._ops.silu_mul_merged_quantize_fp8_static_bf16(gate_up, scale, out)
        return out

    def gelu_mul_merged_quantize_fp8_static_bf16(self, gate_up, scale, out=None):
        if out is None:
            out = torch.empty(
                (gate_up.shape[0], gate_up.shape[1] // 2),
                device=gate_up.device,
                dtype=fp8_dtype(),
            )
        self._ops.gelu_mul_merged_quantize_fp8_static_bf16(gate_up, scale, out)
        return out

    def fp8_swiglu_mlp_bf16(
        self,
        x,
        gate_up_w,
        down_w,
        x_scale,
        gate_up_w_scale,
        hidden_scale,
        down_w_scale,
        gate_up=None,
        hidden_fp8=None,
        out=None,
    ):
        if gate_up is None:
            gate_up = torch.empty(
                (x.shape[0], gate_up_w.shape[0]),
                device=x.device,
                dtype=torch.bfloat16,
            )
        if hidden_fp8 is None:
            hidden_fp8 = torch.empty(
                (x.shape[0], gate_up_w.shape[0] // 2),
                device=x.device,
                dtype=fp8_dtype(),
            )
        if out is None:
            out = torch.empty((x.shape[0], down_w.shape[0]), device=x.device, dtype=torch.bfloat16)
        self._ops.fp8_swiglu_mlp_bf16(
            x,
            gate_up_w,
            down_w,
            x_scale,
            gate_up_w_scale,
            hidden_scale,
            down_w_scale,
            gate_up,
            hidden_fp8,
            out,
        )
        return out

    def fp8_geglu_mlp_bf16(
        self,
        x,
        gate_up_w,
        down_w,
        x_scale,
        gate_up_w_scale,
        hidden_scale,
        down_w_scale,
        gate_up=None,
        hidden_fp8=None,
        out=None,
    ):
        if gate_up is None:
            gate_up = torch.empty(
                (x.shape[0], gate_up_w.shape[0]),
                device=x.device,
                dtype=torch.bfloat16,
            )
        if hidden_fp8 is None:
            hidden_fp8 = torch.empty(
                (x.shape[0], gate_up_w.shape[0] // 2),
                device=x.device,
                dtype=fp8_dtype(),
            )
        if out is None:
            out = torch.empty((x.shape[0], down_w.shape[0]), device=x.device, dtype=torch.bfloat16)
        self._ops.fp8_geglu_mlp_bf16(
            x,
            gate_up_w,
            down_w,
            x_scale,
            gate_up_w_scale,
            hidden_scale,
            down_w_scale,
            gate_up,
            hidden_fp8,
            out,
        )
        return out

    def _bf16_fp8_glu_mlp_bf16(
        self, op, x, gate_up_w, down_w, x_scale, gate_up_w_scale,
        hidden_scale, down_w_scale, input_fp8=None, gate_up_bf16=None,
        hidden_fp8=None, out=None, *, pad_to=None
    ):
        padded_m = x.shape[0] if pad_to is None else pad_to
        hidden = gate_up_w.shape[0] // 2
        if input_fp8 is None:
            input_fp8 = torch.empty(
                (padded_m, x.shape[1]), device=x.device, dtype=fp8_dtype()
            )
        if gate_up_bf16 is None:
            gate_up_bf16 = torch.empty(
                (padded_m, gate_up_w.shape[0]), device=x.device,
                dtype=torch.bfloat16,
            )
        if hidden_fp8 is None:
            hidden_fp8 = torch.empty(
                (padded_m, hidden), device=x.device, dtype=fp8_dtype()
            )
        if out is None:
            out = torch.empty(
                (padded_m, down_w.shape[0]), device=x.device,
                dtype=torch.bfloat16,
            )
        op(
            x, gate_up_w, down_w, x_scale, gate_up_w_scale, hidden_scale,
            down_w_scale, input_fp8, gate_up_bf16, hidden_fp8, out
        )
        return out[: x.shape[0]]

    def bf16_fp8_swiglu_mlp_bf16(self, *args, **kwargs):
        return self._bf16_fp8_glu_mlp_bf16(
            self._ops.bf16_fp8_swiglu_mlp_bf16, *args, **kwargs
        )

    def bf16_fp8_geglu_mlp_bf16(self, *args, **kwargs):
        return self._bf16_fp8_glu_mlp_bf16(
            self._ops.bf16_fp8_geglu_mlp_bf16, *args, **kwargs
        )


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
    namespace = "flashrt_fp8_swiglu_ffn_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "fp8_swiglu_ffn.cu"),
        ],
        extra_include_paths=[str(PACKAGE / "csrc"), str(REGISTRATION_INCLUDE)],
        extra_cflags=["-O3", "-DCUDA_KERNEL"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-DCUDA_KERNEL"],
        extra_ldflags=["-lcublasLt", "-lcublas"],
        verbose=False,
    )
    return SourceOps(namespace)


def load_installed_ops(artifact: str | None):
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return importlib.import_module("flashrt_fp8_swiglu_ffn")
    finally:
        if artifact:
            sys.path.remove(artifact)


def quantize_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x.float() / scale.float(), -fp8_max(), fp8_max()).to(fp8_dtype())


def quantize_fp8_reciprocal(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Match FlashRT's production static-quant arithmetic order exactly."""
    inv_scale = 1.0 / scale.float()
    return torch.clamp(x.float() * inv_scale, -fp8_max(), fp8_max()).to(fp8_dtype())


def dequant_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x.float() * scale.float()


def ref_gemm(x, w, x_scale, w_scale) -> torch.Tensor:
    return (dequant_fp8(x, x_scale) @ dequant_fp8(w, w_scale).T).to(torch.bfloat16)


def ref_swiglu_quant(gate_up_bf16, hidden_scale) -> torch.Tensor:
    gate, up = gate_up_bf16.float().chunk(2, dim=1)
    hidden = torch.nn.functional.silu(gate) * up
    return quantize_fp8(hidden, hidden_scale)


def ref_geglu_quant(gate_up_bf16, hidden_scale) -> torch.Tensor:
    gate, up = gate_up_bf16.float().chunk(2, dim=1)
    hidden = torch.nn.functional.gelu(gate, approximate="tanh") * up
    return quantize_fp8(hidden, hidden_scale)


def ref_mlp(x, gate_up_w, down_w, x_scale, gate_up_w_scale, hidden_scale, down_w_scale):
    gate_up = ref_gemm(x, gate_up_w, x_scale, gate_up_w_scale)
    hidden_fp8 = ref_swiglu_quant(gate_up, hidden_scale)
    return ref_gemm(hidden_fp8, down_w, hidden_scale, down_w_scale)


def ref_geglu_mlp(x, gate_up_w, down_w, x_scale, gate_up_w_scale, hidden_scale, down_w_scale):
    gate_up = ref_gemm(x, gate_up_w, x_scale, gate_up_w_scale)
    hidden_fp8 = ref_geglu_quant(gate_up, hidden_scale)
    return ref_gemm(hidden_fp8, down_w, hidden_scale, down_w_scale)


def make_case(M: int, K: int, H: int, N: int):
    x_scale = torch.tensor([0.05], device="cuda", dtype=torch.float32)
    gate_up_w_scale = torch.tensor([0.04], device="cuda", dtype=torch.float32)
    hidden_scale = torch.tensor([0.25], device="cuda", dtype=torch.float32)
    down_w_scale = torch.tensor([0.04], device="cuda", dtype=torch.float32)
    x = quantize_fp8(torch.randn((M, K), device="cuda", dtype=torch.bfloat16), x_scale)
    gate_up_w = quantize_fp8(
        torch.randn((2 * H, K), device="cuda", dtype=torch.bfloat16),
        gate_up_w_scale,
    )
    down_w = quantize_fp8(
        torch.randn((N, H), device="cuda", dtype=torch.bfloat16),
        down_w_scale,
    )
    return x, gate_up_w, down_w, x_scale, gate_up_w_scale, hidden_scale, down_w_scale


def percentile(x: torch.Tensor, q: float) -> torch.Tensor:
    flat = x.flatten()
    k = max(1, min(flat.numel(), math.ceil(q * flat.numel())))
    return flat.kthvalue(k).values


def distribution_metrics(got: torch.Tensor, expected: torch.Tensor):
    diff = (got.float() - expected.float()).abs().flatten()
    exp = expected.float().abs().flatten().clamp_min(1.0)
    rel = diff / exp
    got_f = got.float().flatten()
    exp_f = expected.float().flatten()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "p99_abs": float(percentile(diff, 0.99).item()),
        "max_rel": float(rel.max().item()),
        "p99_rel": float(percentile(rel, 0.99).item()),
        "cosine": float(torch.nn.functional.cosine_similarity(got_f, exp_f, dim=0).item()),
    }


def assert_distribution_close(
    name: str,
    got: torch.Tensor,
    expected: torch.Tensor,
    *,
    p99_abs_limit: float,
    p99_rel_floor1_limit: float,
    max_abs_limit: float | None = None,
    cosine_min: float = 0.0,
) -> None:
    m = distribution_metrics(got, expected)
    if (
        m["p99_abs"] > p99_abs_limit
        or m["p99_rel"] > p99_rel_floor1_limit
        or (max_abs_limit is not None and m["max_abs"] > max_abs_limit)
        or m["cosine"] < cosine_min
    ):
        raise AssertionError(
            f"{name} failed: max_abs={m['max_abs']} p99_abs={m['p99_abs']} "
            f"p99_rel_floor1={m['p99_rel']} max_rel_floor1={m['max_rel']}"
        )
    print(
        f"PASS {name}: max_abs={m['max_abs']:.6f} "
        f"mean_abs={m['mean_abs']:.6f} p99_abs={m['p99_abs']:.6f} "
        f"cosine={m['cosine']:.8f} p99_rel_floor1={m['p99_rel']:.6f} "
        f"max_rel_floor1={m['max_rel']:.6f}"
    )


def report_distribution(name: str, got: torch.Tensor, expected: torch.Tensor) -> None:
    m = distribution_metrics(got, expected)
    print(
        f"INFO {name}: max_abs={m['max_abs']:.6f} "
        f"mean_abs={m['mean_abs']:.6f} p99_abs={m['p99_abs']:.6f} "
        f"cosine={m['cosine']:.8f} p99_rel_floor1={m['p99_rel']:.6f} "
        f"max_rel_floor1={m['max_rel']:.6f}"
    )


def assert_fp8_quant_close(
    name: str,
    got: torch.Tensor,
    expected: torch.Tensor,
    *,
    p99_abs_limit: float,
    mismatch_rate_limit: float,
) -> None:
    diff = (got.float() - expected.float()).abs().flatten()
    mismatches = int((got.detach().cpu() != expected.detach().cpu()).sum().item())
    mismatch_rate = mismatches / got.numel()
    max_abs = float(diff.max().item())
    p99_abs = float(percentile(diff, 0.99).item())
    if p99_abs > p99_abs_limit or mismatch_rate > mismatch_rate_limit:
        raise AssertionError(
            f"{name} failed: fp8_max_abs={max_abs} fp8_p99_abs={p99_abs} "
            f"mismatches={mismatches} mismatch_rate={mismatch_rate}"
        )
    print(
        f"PASS {name}: fp8_max_abs={max_abs:.6f} fp8_p99_abs={p99_abs:.6f} "
        f"mismatches={mismatches} mismatch_rate={mismatch_rate:.8f}"
    )


def expect_runtime_error(label: str, fn) -> None:
    try:
        fn()
    except RuntimeError as exc:
        print(f"PASS {label}: rejected invalid input ({str(exc).splitlines()[0]})")
        return
    raise AssertionError(f"{label} failed: expected RuntimeError")


def run_shape(ops, label: str, shape: tuple[int, int, int, int]) -> None:
    x, gate_up_w, down_w, x_s, gu_s, hid_s, dn_s = make_case(*shape)

    got_gemm = ops.fp8_gemm_bf16(x, gate_up_w, x_s, gu_s)
    exp_gemm = ref_gemm(x, gate_up_w, x_s, gu_s)
    assert_distribution_close(
        f"{label}/fp8_gemm_bf16",
        got_gemm,
        exp_gemm,
        p99_abs_limit=0.25,
        p99_rel_floor1_limit=0.03,
        max_abs_limit=1.0,
    )

    got_hidden_fp8 = ops.silu_mul_merged_quantize_fp8_static_bf16(got_gemm, hid_s)
    exp_hidden_fp8 = ref_swiglu_quant(exp_gemm, hid_s)
    # SiLU/GELU use transcendental functions; CUDA libdevice and PyTorch eager
    # can land on opposite sides of an FP8 bin boundary for a small fraction of
    # values. The production migration invariant is the exact fused-vs-staged
    # check below; this reference check is distributional.
    assert_fp8_quant_close(
        f"{label}/silu_mul_merged_quantize_fp8_static_bf16",
        got_hidden_fp8,
        exp_hidden_fp8,
        p99_abs_limit=1.0,
        mismatch_rate_limit=0.03,
    )

    staged_out = ops.fp8_gemm_bf16(got_hidden_fp8, down_w, hid_s, dn_s)
    got_mlp = ops.fp8_swiglu_mlp_bf16(x, gate_up_w, down_w, x_s, gu_s, hid_s, dn_s)
    assert_distribution_close(
        f"{label}/fp8_swiglu_mlp_bf16_vs_staged_ops",
        got_mlp,
        staged_out,
        p99_abs_limit=0.0,
        p99_rel_floor1_limit=0.0,
        max_abs_limit=0.0,
    )
    report_distribution(
        f"{label}/fp8_swiglu_mlp_bf16_vs_torch_reference",
        got_mlp,
        ref_mlp(x, gate_up_w, down_w, x_s, gu_s, hid_s, dn_s),
    )

    got_geglu_hidden_fp8 = ops.gelu_mul_merged_quantize_fp8_static_bf16(got_gemm, hid_s)
    exp_geglu_hidden_fp8 = ref_geglu_quant(exp_gemm, hid_s)
    assert_fp8_quant_close(
        f"{label}/gelu_mul_merged_quantize_fp8_static_bf16",
        got_geglu_hidden_fp8,
        exp_geglu_hidden_fp8,
        p99_abs_limit=1.0,
        mismatch_rate_limit=0.03,
    )

    staged_geglu_out = ops.fp8_gemm_bf16(got_geglu_hidden_fp8, down_w, hid_s, dn_s)
    got_geglu_mlp = ops.fp8_geglu_mlp_bf16(x, gate_up_w, down_w, x_s, gu_s, hid_s, dn_s)
    assert_distribution_close(
        f"{label}/fp8_geglu_mlp_bf16_vs_staged_ops",
        got_geglu_mlp,
        staged_geglu_out,
        p99_abs_limit=0.0,
        p99_rel_floor1_limit=0.0,
        max_abs_limit=0.0,
    )
    report_distribution(
        f"{label}/fp8_geglu_mlp_bf16_vs_torch_reference",
        got_geglu_mlp,
        ref_geglu_mlp(x, gate_up_w, down_w, x_s, gu_s, hid_s, dn_s),
    )


def run_rejection_tests(ops) -> None:
    x, gate_up_w, down_w, x_s, gu_s, hid_s, dn_s = make_case(4, 128, 256, 128)
    bad_gate_up = torch.empty((4, 511), device="cuda", dtype=torch.bfloat16)
    bad_hidden = torch.empty((4, 255), device="cuda", dtype=fp8_dtype())

    expect_runtime_error(
        "reject odd gate_up columns",
        lambda: ops.silu_mul_merged_quantize_fp8_static_bf16(bad_gate_up, hid_s),
    )
    expect_runtime_error(
        "reject invalid hidden_fp8 shape",
        lambda: ops.fp8_swiglu_mlp_bf16(
            x,
            gate_up_w,
            down_w,
            x_s,
            gu_s,
            hid_s,
            dn_s,
            hidden_fp8=bad_hidden,
        ),
    )


def run(args) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(11)
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)

    shapes = {
        "small": (16, 128, 256, 128),
        "pi05_decoder_m10": (10, 1024, 4096, 1024),
        "pi05_vision_2view": (512, 1152, 4304, 1152),
        "groot_vl_seq512": (512, 2048, 8192, 2048),
        "groot_action_dit": (41, 1536, 6144, 1536),
    }
    if args.mode == "smoke":
        shapes = {k: shapes[k] for k in ("small", "pi05_decoder_m10")}

    for label, shape in shapes.items():
        run_shape(ops, label, shape)

    for label, shape in {
        "midm_decoder_m8": (8, 1024, 4096, 1024),
        "midm_decoder_m51": (51, 1024, 4096, 1024),
        "midm_decoder_m64": (64, 1024, 4096, 1024),
        "midm_decoder_m105": (105, 1024, 4096, 1024),
        "midm_decoder_m128": (128, 1024, 4096, 1024),
        "midm_dit_m51": (51, 1536, 6144, 1536),
    }.items():
        M, K, H, N = shape
        x_bf16 = torch.randn((M, K), device="cuda", dtype=torch.bfloat16) * 0.25
        x_scale = (x_bf16.float().abs().max() / (0.9 * fp8_max())).clamp_min(1e-6).reshape(1)
        x_fp8 = quantize_fp8_reciprocal(x_bf16, x_scale)
        _, gate_up_w, down_w, _, gu_s, hid_s, dn_s = make_case(*shape)
        padded_m = midm_padded_rows(M)
        input_fp8 = torch.empty(
            (padded_m, K), device="cuda", dtype=fp8_dtype()
        )
        got = ops.bf16_fp8_swiglu_mlp_bf16(
            x_bf16, gate_up_w, down_w, x_scale, gu_s, hid_s, dn_s,
            input_fp8=input_fp8, pad_to=padded_m,
        )
        staged = ops.fp8_swiglu_mlp_bf16(
            input_fp8, gate_up_w, down_w, x_scale, gu_s, hid_s, dn_s
        )[:M]
        assert_distribution_close(
            f"{label}/bf16_input_quant", input_fp8[:M], x_fp8,
            p99_abs_limit=0.0, p99_rel_floor1_limit=0.0,
            max_abs_limit=0.0,
        )
        if padded_m > M:
            assert_distribution_close(
                f"{label}/bf16_input_padding",
                input_fp8[M:],
                torch.zeros_like(input_fp8[M:]),
                p99_abs_limit=0.0,
                p99_rel_floor1_limit=0.0,
                max_abs_limit=0.0,
            )
        assert_distribution_close(
            f"{label}/bf16_fp8_swiglu_mlp_vs_staged", got, staged,
            p99_abs_limit=0.0, p99_rel_floor1_limit=0.0,
            max_abs_limit=0.0,
        )
        if M == 51:
            got_geglu = ops.bf16_fp8_geglu_mlp_bf16(
                x_bf16, gate_up_w, down_w, x_scale, gu_s, hid_s, dn_s,
                input_fp8=input_fp8, pad_to=padded_m,
            )
            staged_geglu = ops.fp8_geglu_mlp_bf16(
                input_fp8, gate_up_w, down_w, x_scale, gu_s, hid_s, dn_s
            )[:M]
            assert_distribution_close(
                f"{label}/bf16_fp8_geglu_mlp_vs_staged", got_geglu,
                staged_geglu, p99_abs_limit=0.0,
                p99_rel_floor1_limit=0.0, max_abs_limit=0.0,
            )

    run_rejection_tests(ops)

    M, K, H, N = 51, 128, 256, 128
    x_bf16 = torch.randn((M, K), device="cuda", dtype=torch.bfloat16) * 0.25
    x_scale = torch.tensor([0.01], device="cuda", dtype=torch.float32)
    gate_up_scale = torch.tensor([0.02], device="cuda", dtype=torch.float32)
    hidden_scale = torch.tensor([0.02], device="cuda", dtype=torch.float32)
    down_scale = torch.tensor([0.02], device="cuda", dtype=torch.float32)
    gate_up_weight = quantize_fp8(
        torch.randn((2 * H, K), device="cuda", dtype=torch.bfloat16) * 0.02,
        gate_up_scale,
    )
    down_weight = quantize_fp8(
        torch.randn((N, H), device="cuda", dtype=torch.bfloat16) * 0.02,
        down_scale,
    )
    padded_m = 64
    input_fp8 = torch.empty((padded_m, K), device="cuda", dtype=fp8_dtype())
    gate_up_bf16 = torch.empty((padded_m, 2 * H), device="cuda", dtype=torch.bfloat16)
    hidden_fp8 = torch.empty((padded_m, H), device="cuda", dtype=fp8_dtype())
    out = torch.empty((padded_m, N), device="cuda", dtype=torch.bfloat16)

    def region(value):
        return ops.bf16_fp8_swiglu_mlp_bf16(
            value, gate_up_weight, down_weight, x_scale, gate_up_scale,
            hidden_scale, down_scale, input_fp8=input_fp8,
            gate_up_bf16=gate_up_bf16, hidden_fp8=hidden_fp8, out=out,
            pad_to=padded_m,
        )

    expected = region(x_bf16).clone()
    compiled = torch.compile(region, fullgraph=True)
    assert_distribution_close(
        "bf16_region/torch_compile_fullgraph", compiled(x_bf16), expected,
        p99_abs_limit=0.0, p99_rel_floor1_limit=0.0, max_abs_limit=0.0,
    )
    graph = torch.cuda.CUDAGraph()
    region(x_bf16)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        region(x_bf16)
    graph.replay()
    torch.cuda.synchronize()
    assert_distribution_close(
        "bf16_region/cuda_graph_replay", out[:M], expected,
        p99_abs_limit=0.0, p99_rel_floor1_limit=0.0, max_abs_limit=0.0,
    )
    assert_distribution_close(
        "bf16_region/padded_input_zero_fill",
        input_fp8[M:],
        torch.zeros_like(input_fp8[M:]),
        p99_abs_limit=0.0,
        p99_rel_floor1_limit=0.0,
        max_abs_limit=0.0,
    )
    if out.dtype != torch.bfloat16:
        raise AssertionError(f"output dtype must be bfloat16, got {out.dtype}")
    print("PASS bf16_region/output_dtype: torch.bfloat16")

    try:
        ops.bf16_fp8_swiglu_mlp_bf16(
            x_bf16, gate_up_weight, down_weight, x_scale, gate_up_scale,
            hidden_scale, down_scale,
            input_fp8=torch.empty((M - 1, K), device="cuda", dtype=fp8_dtype()),
            pad_to=M - 1,
        )
    except (RuntimeError, ValueError):
        print("PASS bf16_region/reject_short_padding")
    else:
        raise AssertionError("short padded scratch must be rejected")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
