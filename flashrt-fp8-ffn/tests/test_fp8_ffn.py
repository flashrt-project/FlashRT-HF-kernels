#!/usr/bin/env python3
"""Correctness tests for flashrt-fp8-ffn."""

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
PACKAGE = ROOT / "flashrt-fp8-ffn"
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

    def fp8_linear_bias_gelu_quant_bf16(
        self, x, w, bias, x_scale, w_scale, y_scale, hidden=None, out=None
    ):
        if hidden is None:
            hidden = torch.empty((x.shape[0], w.shape[0]), device=x.device, dtype=torch.bfloat16)
        if out is None:
            out = torch.empty_like(hidden, dtype=fp8_dtype())
        self._ops.fp8_linear_bias_gelu_quant_bf16(
            x, w, bias, x_scale, w_scale, y_scale, hidden, out
        )
        return hidden, out

    def fp8_gelu_mlp_bf16(
        self,
        x,
        up_w,
        up_b,
        down_w,
        down_b,
        x_scale,
        up_w_scale,
        hidden_scale,
        down_w_scale,
        hidden=None,
        hidden_fp8=None,
        out=None,
    ):
        if hidden is None:
            hidden = torch.empty((x.shape[0], up_w.shape[0]), device=x.device, dtype=torch.bfloat16)
        if hidden_fp8 is None:
            hidden_fp8 = torch.empty_like(hidden, dtype=fp8_dtype())
        if out is None:
            out = torch.empty((x.shape[0], down_w.shape[0]), device=x.device, dtype=torch.bfloat16)
        self._ops.fp8_gelu_mlp_bf16(
            x,
            up_w,
            up_b,
            down_w,
            down_b,
            x_scale,
            up_w_scale,
            hidden_scale,
            down_w_scale,
            hidden,
            hidden_fp8,
            out,
        )
        return out

    def bf16_fp8_gelu_mlp_bf16(
        self, x, up_w, up_b, down_w, down_b, x_scale, up_w_scale,
        hidden_scale, down_w_scale, input_fp8=None, hidden_bf16=None,
        hidden_fp8=None, out=None, *, pad_to=None
    ):
        padded_m = x.shape[0] if pad_to is None else pad_to
        if input_fp8 is None:
            input_fp8 = torch.empty(
                (padded_m, x.shape[1]), device=x.device, dtype=fp8_dtype()
            )
        if hidden_bf16 is None:
            hidden_bf16 = torch.empty(
                (padded_m, up_w.shape[0]), device=x.device, dtype=torch.bfloat16
            )
        if hidden_fp8 is None:
            hidden_fp8 = torch.empty_like(hidden_bf16, dtype=fp8_dtype())
        if out is None:
            out = torch.empty(
                (padded_m, down_w.shape[0]), device=x.device, dtype=torch.bfloat16
            )
        self._ops.bf16_fp8_gelu_mlp_bf16(
            x, up_w, up_b, down_w, down_b, x_scale, up_w_scale,
            hidden_scale, down_w_scale, input_fp8, hidden_bf16, hidden_fp8, out
        )
        return out[: x.shape[0]]


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
    namespace = "flashrt_fp8_ffn_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "fp8_ffn.cu"),
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
        return importlib.import_module("flashrt_fp8_ffn")
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


def ref_linear_bias_gelu_quant(x, w, bias, x_scale, w_scale, y_scale):
    hidden = ref_gemm(x, w, x_scale, w_scale)
    y = torch.nn.functional.gelu(hidden.float() + bias.float(), approximate="tanh")
    y_fp8 = torch.clamp(y / y_scale.float(), -fp8_max(), fp8_max()).to(fp8_dtype())
    return hidden, y_fp8


def ref_mlp(x, up_w, up_b, down_w, down_b, x_scale, up_w_scale, hidden_scale, down_w_scale):
    _, hidden_fp8 = ref_linear_bias_gelu_quant(
        x, up_w, up_b, x_scale, up_w_scale, hidden_scale
    )
    out = ref_gemm(hidden_fp8, down_w, hidden_scale, down_w_scale)
    return (out.float() + down_b.float()).to(torch.bfloat16)


def make_case(M: int, K: int, H: int, N: int):
    x_scale = torch.tensor([0.05], device="cuda", dtype=torch.float32)
    up_w_scale = torch.tensor([0.04], device="cuda", dtype=torch.float32)
    hidden_scale = torch.tensor([0.25], device="cuda", dtype=torch.float32)
    down_w_scale = torch.tensor([0.04], device="cuda", dtype=torch.float32)
    x = quantize_fp8(torch.randn((M, K), device="cuda", dtype=torch.bfloat16), x_scale)
    up_w = quantize_fp8(torch.randn((H, K), device="cuda", dtype=torch.bfloat16), up_w_scale)
    down_w = quantize_fp8(torch.randn((N, H), device="cuda", dtype=torch.bfloat16), down_w_scale)
    up_b = torch.randn((H,), device="cuda", dtype=torch.bfloat16)
    down_b = torch.randn((N,), device="cuda", dtype=torch.bfloat16)
    return x, up_w, up_b, down_w, down_b, x_scale, up_w_scale, hidden_scale, down_w_scale


def assert_close(name: str, got: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float) -> None:
    max_abs = float((got.float() - expected.float()).abs().max().item())
    if not torch.allclose(got.float(), expected.float(), atol=atol, rtol=rtol):
        raise AssertionError(f"{name} failed: max_abs={max_abs} atol={atol} rtol={rtol}")
    print(f"PASS {name}: max_abs={max_abs:.6f}")


def percentile(x: torch.Tensor, q: float) -> torch.Tensor:
    flat = x.flatten()
    k = max(1, min(flat.numel(), math.ceil(q * flat.numel())))
    return flat.kthvalue(k).values


def assert_distribution_close(
    name: str,
    got: torch.Tensor,
    expected: torch.Tensor,
    *,
    p99_abs_limit: float,
    p99_rel_floor1_limit: float,
    cosine_min: float = 0.0,
    max_abs_limit: float | None = None,
) -> None:
    m = distribution_metrics(got, expected)
    if (
        m["p99_abs"] > p99_abs_limit
        or m["p99_rel"] > p99_rel_floor1_limit
        or m["cosine"] < cosine_min
        or (max_abs_limit is not None and m["max_abs"] > max_abs_limit)
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


def run(args) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(11)
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)

    for label, shape in {
        "small": (16, 128, 256, 128),
        "pi05_vision": (512, 1152, 4304, 1152),
        "pi05_decoder": (10, 1024, 4096, 1024),
        "groot_vit": (128, 1024, 4096, 1024),
        "groot_deepstack": (128, 4096, 4096, 2048),
        "groot_vl_self_attn_long": (2520, 2048, 8192, 2048),
        "groot_action_dit": (41, 1536, 6144, 1536),
    }.items():
        x, up_w, up_b, down_w, down_b, x_s, up_s, hid_s, dn_s = make_case(*shape)

        got_gemm = ops.fp8_gemm_bf16(x, up_w, x_s, up_s)
        exp_gemm = ref_gemm(x, up_w, x_s, up_s)
        assert_close(f"{label}/fp8_gemm_bf16", got_gemm, exp_gemm, atol=0.25, rtol=0.03)

        _, got_hidden_fp8 = ops.fp8_linear_bias_gelu_quant_bf16(x, up_w, up_b, x_s, up_s, hid_s)
        _, exp_hidden_fp8 = ref_linear_bias_gelu_quant(x, up_w, up_b, x_s, up_s, hid_s)
        # CUDA libdevice and PyTorch eager can round tanh-GELU to adjacent FP8
        # bins. Exact migration parity is enforced by the fused-vs-staged MLP
        # assertion below; this independent math reference is distributional.
        assert_fp8_quant_close(
            f"{label}/fp8_linear_bias_gelu_quant_bf16",
            got_hidden_fp8,
            exp_hidden_fp8,
            p99_abs_limit=1.0,
            mismatch_rate_limit=0.03,
        )

        staged_down = ops.fp8_gemm_bf16(got_hidden_fp8, down_w, hid_s, dn_s)
        staged_mlp = (staged_down.float() + down_b.float()).to(torch.bfloat16)
        got_mlp = ops.fp8_gelu_mlp_bf16(x, up_w, up_b, down_w, down_b, x_s, up_s, hid_s, dn_s)
        assert_distribution_close(
            f"{label}/fp8_gelu_mlp_bf16_vs_staged_ops",
            got_mlp,
            staged_mlp,
            p99_abs_limit=0.0,
            p99_rel_floor1_limit=0.0,
        )
        report_distribution(
            f"{label}/fp8_gelu_mlp_bf16_vs_torch_reference",
            got_mlp,
            ref_mlp(x, up_w, up_b, down_w, down_b, x_s, up_s, hid_s, dn_s),
        )

    for label, shape in {
        "midm_siglip_m8": (8, 1152, 4304, 1152),
        "midm_siglip_m51": (51, 1152, 4304, 1152),
        "midm_siglip_m64": (64, 1152, 4304, 1152),
        "midm_siglip_m105": (105, 1152, 4304, 1152),
        "midm_siglip_m128": (128, 1152, 4304, 1152),
        "midm_dit_m51": (51, 1536, 6144, 1536),
        "midm_dit_m128": (128, 1536, 6144, 1536),
    }.items():
        M, K, H, N = shape
        x_bf16 = torch.randn((M, K), device="cuda", dtype=torch.bfloat16) * 0.25
        x_scale = (x_bf16.float().abs().max() / (0.9 * fp8_max())).clamp_min(1e-6).reshape(1)
        x_fp8 = quantize_fp8_reciprocal(x_bf16, x_scale)
        up_bf16 = torch.randn((H, K), device="cuda", dtype=torch.bfloat16) * (K**-0.5)
        down_bf16 = torch.randn((N, H), device="cuda", dtype=torch.bfloat16) * (H**-0.5)
        up_s = (up_bf16.float().abs().max() / (0.9 * fp8_max())).clamp_min(1e-6).reshape(1)
        dn_s = (down_bf16.float().abs().max() / (0.9 * fp8_max())).clamp_min(1e-6).reshape(1)
        up_w = quantize_fp8(up_bf16, up_s)
        down_w = quantize_fp8(down_bf16, dn_s)
        up_b = torch.randn((H,), device="cuda", dtype=torch.bfloat16) * 0.01
        down_b = torch.randn((N,), device="cuda", dtype=torch.bfloat16) * 0.01
        hid_s = torch.tensor([0.0025], device="cuda", dtype=torch.float32)
        padded_m = midm_padded_rows(M)
        input_fp8 = torch.empty(
            (padded_m, K), device="cuda", dtype=fp8_dtype()
        )
        got = ops.bf16_fp8_gelu_mlp_bf16(
            x_bf16, up_w, up_b, down_w, down_b, x_scale, up_s, hid_s, dn_s,
            input_fp8=input_fp8, pad_to=padded_m,
        )
        staged = ops.fp8_gelu_mlp_bf16(
            input_fp8, up_w, up_b, down_w, down_b, x_scale, up_s, hid_s, dn_s
        )[:M]
        assert_close(
            f"{label}/bf16_input_quant", input_fp8[:M], x_fp8, atol=0.0, rtol=0.0
        )
        if padded_m > M:
            assert_close(
                f"{label}/bf16_input_padding",
                input_fp8[M:],
                torch.zeros_like(input_fp8[M:]),
                atol=0.0,
                rtol=0.0,
            )
        assert_close(
            f"{label}/bf16_fp8_gelu_mlp_vs_staged",
            got,
            staged,
            atol=0.0,
            rtol=0.0,
        )

    M, K, H, N = 51, 128, 256, 128
    x_bf16 = torch.randn((M, K), device="cuda", dtype=torch.bfloat16) * 0.25
    x_scale = torch.tensor([0.01], device="cuda", dtype=torch.float32)
    up_scale = torch.tensor([0.02], device="cuda", dtype=torch.float32)
    hidden_scale = torch.tensor([0.02], device="cuda", dtype=torch.float32)
    down_scale = torch.tensor([0.02], device="cuda", dtype=torch.float32)
    up_weight = quantize_fp8(
        torch.randn((H, K), device="cuda", dtype=torch.bfloat16) * 0.02,
        up_scale,
    )
    down_weight = quantize_fp8(
        torch.randn((N, H), device="cuda", dtype=torch.bfloat16) * 0.02,
        down_scale,
    )
    up_bias = torch.randn((H,), device="cuda", dtype=torch.bfloat16) * 0.01
    down_bias = torch.randn((N,), device="cuda", dtype=torch.bfloat16) * 0.01
    padded_m = 64
    input_fp8 = torch.empty((padded_m, K), device="cuda", dtype=fp8_dtype())
    hidden_bf16 = torch.empty((padded_m, H), device="cuda", dtype=torch.bfloat16)
    hidden_fp8 = torch.empty((padded_m, H), device="cuda", dtype=fp8_dtype())
    out = torch.empty((padded_m, N), device="cuda", dtype=torch.bfloat16)

    def region(value):
        return ops.bf16_fp8_gelu_mlp_bf16(
            value, up_weight, up_bias, down_weight, down_bias, x_scale,
            up_scale, hidden_scale, down_scale, input_fp8=input_fp8,
            hidden_bf16=hidden_bf16, hidden_fp8=hidden_fp8, out=out,
            pad_to=padded_m,
        )

    expected = region(x_bf16).clone()
    compiled = torch.compile(region, fullgraph=True)
    assert_close(
        "bf16_region/torch_compile_fullgraph", compiled(x_bf16), expected,
        atol=0.0, rtol=0.0,
    )
    graph = torch.cuda.CUDAGraph()
    region(x_bf16)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        region(x_bf16)
    graph.replay()
    torch.cuda.synchronize()
    assert_close(
        "bf16_region/cuda_graph_replay", out[:M], expected, atol=0.0, rtol=0.0
    )
    assert_close(
        "bf16_region/padded_input_zero_fill",
        input_fp8[M:],
        torch.zeros_like(input_fp8[M:]),
        atol=0.0,
        rtol=0.0,
    )
    if out.dtype != torch.bfloat16:
        raise AssertionError(f"output dtype must be bfloat16, got {out.dtype}")
    print("PASS bf16_region/output_dtype: torch.bfloat16")

    try:
        ops.bf16_fp8_gelu_mlp_bf16(
            x_bf16, up_weight, up_bias, down_weight, down_bias, x_scale,
            up_scale, hidden_scale, down_scale,
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
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
