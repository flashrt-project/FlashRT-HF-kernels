#!/usr/bin/env python3
"""Benchmark flashrt-fp8-swiglu-ffn against PyTorch references."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import importlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
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

SHAPES = {
    "pi05_decoder_m1": (1, 1024, 4096, 1024),
    "pi05_decoder_m8": (8, 1024, 4096, 1024),
    "pi05_decoder_m10": (10, 1024, 4096, 1024),
    "pi05_decoder_m16": (16, 1024, 4096, 1024),
    "pi05_vision_1view": (256, 1152, 4304, 1152),
    "pi05_vision_2view": (512, 1152, 4304, 1152),
    "pi05_vision_3view": (768, 1152, 4304, 1152),
    "groot_vl_seq512": (512, 2048, 8192, 2048),
    "groot_vl_seq1024": (1024, 2048, 8192, 2048),
    "groot_vl_seq2520": (2520, 2048, 8192, 2048),
    "action_dit": (41, 1536, 6144, 1536),
}

SHAPE_GROUPS = {
    "smoke": ["pi05_decoder_m10"],
    "headline": ["pi05_decoder_m10", "pi05_vision_2view", "groot_vl_seq1024"],
    "pi05": [
        "pi05_decoder_m1",
        "pi05_decoder_m8",
        "pi05_decoder_m10",
        "pi05_decoder_m16",
        "pi05_vision_1view",
        "pi05_vision_2view",
        "pi05_vision_3view",
    ],
    "all": list(SHAPES.keys()),
}


@dataclass
class Result:
    shape: str
    M: int
    K: int
    H: int
    N: int
    flashrt_us: float
    torch_eager_us: float
    torch_compile_us: float | None
    speedup_vs_eager: float
    speedup_vs_compile: float | None
    compile_status: str
    max_abs: float
    mean_abs: float
    p99_abs: float
    cosine: float
    p99_rel_floor1: float
    torch_ref_max_abs: float
    torch_ref_mean_abs: float
    torch_ref_p99_abs: float
    torch_ref_cosine: float
    torch_ref_p99_rel_floor1: float
    status: str


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

    def fp8_swiglu_mlp_bf16(
        self,
        x,
        gate_up_w,
        down_w,
        x_scale,
        gate_up_w_scale,
        hidden_scale,
        down_w_scale,
        gate_up_bf16=None,
        hidden_fp8=None,
        out=None,
    ):
        if gate_up_bf16 is None:
            gate_up_bf16 = torch.empty(
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
            gate_up_bf16,
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
        gate_up_bf16=None,
        hidden_fp8=None,
        out=None,
    ):
        if gate_up_bf16 is None:
            gate_up_bf16 = torch.empty(
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
            out = torch.empty(
                (x.shape[0], down_w.shape[0]),
                device=x.device,
                dtype=torch.bfloat16,
            )
        self._ops.fp8_geglu_mlp_bf16(
            x,
            gate_up_w,
            down_w,
            x_scale,
            gate_up_w_scale,
            hidden_scale,
            down_w_scale,
            gate_up_bf16,
            hidden_fp8,
            out,
        )
        return out

    def _bf16_fp8_glu_mlp_bf16(
        self,
        op,
        x,
        gate_up_w,
        down_w,
        x_scale,
        gate_up_w_scale,
        hidden_scale,
        down_w_scale,
        input_fp8=None,
        gate_up_bf16=None,
        hidden_fp8=None,
        out=None,
        *,
        pad_to=None,
    ):
        padded_m = x.shape[0] if pad_to is None else pad_to
        hidden = gate_up_w.shape[0] // 2
        if input_fp8 is None:
            input_fp8 = torch.empty(
                (padded_m, x.shape[1]), device=x.device, dtype=fp8_dtype()
            )
        if gate_up_bf16 is None:
            gate_up_bf16 = torch.empty(
                (padded_m, gate_up_w.shape[0]),
                device=x.device,
                dtype=torch.bfloat16,
            )
        if hidden_fp8 is None:
            hidden_fp8 = torch.empty(
                (padded_m, hidden), device=x.device, dtype=fp8_dtype()
            )
        if out is None:
            out = torch.empty(
                (padded_m, down_w.shape[0]),
                device=x.device,
                dtype=torch.bfloat16,
            )
        op(
            x,
            gate_up_w,
            down_w,
            x_scale,
            gate_up_w_scale,
            hidden_scale,
            down_w_scale,
            input_fp8,
            gate_up_bf16,
            hidden_fp8,
            out,
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
    namespace = "flashrt_fp8_swiglu_ffn_benchmark"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "fp8_swiglu_ffn.cu"),
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
        return importlib.import_module("flashrt_fp8_swiglu_ffn")
    finally:
        if artifact:
            sys.path.remove(artifact)


def load_hub_ops(repo_id: str, version: int):
    from kernels import get_kernel

    return get_kernel(repo_id, version=version)


def fp8_dtype() -> torch.dtype:
    if torch.version.hip is not None and hasattr(torch, "float8_e4m3fnuz"):
        return torch.float8_e4m3fnuz
    return torch.float8_e4m3fn


def fp8_max() -> float:
    return 240.0 if torch.version.hip is not None else 448.0


def quantize_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    limit = fp8_max()
    return torch.clamp(x.float() / scale.float(), -limit, limit).to(fp8_dtype())


def dequant_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x.float() * scale.float()


def torch_ref(x, gate_up_w, down_w, x_s, gu_s, hid_s, dn_s):
    gate_up = (dequant_fp8(x, x_s) @ dequant_fp8(gate_up_w, gu_s).T).to(torch.bfloat16)
    gate, up = gate_up.float().chunk(2, dim=1)
    hidden_fp8 = quantize_fp8(torch.nn.functional.silu(gate) * up, hid_s)
    return (dequant_fp8(hidden_fp8, hid_s) @ dequant_fp8(down_w, dn_s).T).to(torch.bfloat16)


def compiler_disable(fn):
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and hasattr(compiler, "disable"):
        return compiler.disable(fn)
    return torch._dynamo.disable(fn)


def swiglu_quant_boundary(gate_up: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up.float().chunk(2, dim=1)
    return quantize_fp8(torch.nn.functional.silu(gate) * up, scale)


stable_swiglu_quant_boundary = compiler_disable(swiglu_quant_boundary)


def ref_fp8_gemm(x, w, x_scale, w_scale):
    return (dequant_fp8(x, x_scale) @ dequant_fp8(w, w_scale).T).to(torch.bfloat16)


def make_case(M: int, K: int, H: int, N: int):
    x_s = torch.tensor([0.05], device="cuda", dtype=torch.float32)
    gu_s = torch.tensor([0.04], device="cuda", dtype=torch.float32)
    hid_s = torch.tensor([0.25], device="cuda", dtype=torch.float32)
    dn_s = torch.tensor([0.04], device="cuda", dtype=torch.float32)
    x = quantize_fp8(torch.randn((M, K), device="cuda", dtype=torch.bfloat16), x_s)
    gate_up_w = quantize_fp8(
        torch.randn((2 * H, K), device="cuda", dtype=torch.bfloat16),
        gu_s,
    )
    down_w = quantize_fp8(torch.randn((N, H), device="cuda", dtype=torch.bfloat16), dn_s)
    scratch_gate_up = torch.empty((M, 2 * H), device="cuda", dtype=torch.bfloat16)
    scratch_hidden = torch.empty((M, H), device="cuda", dtype=fp8_dtype())
    out = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    return x, gate_up_w, down_w, x_s, gu_s, hid_s, dn_s, scratch_gate_up, scratch_hidden, out


def time_us(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def percentile(x: torch.Tensor, q: float) -> torch.Tensor:
    flat = x.flatten()
    k = max(1, min(flat.numel(), math.ceil(q * flat.numel())))
    return flat.kthvalue(k).values


def metrics(got: torch.Tensor, expected: torch.Tensor):
    diff = (got.float() - expected.float()).abs().flatten()
    rel = diff / expected.float().abs().flatten().clamp_min(1.0)
    cosine = torch.nn.functional.cosine_similarity(
        got.float().flatten(), expected.float().flatten(), dim=0
    )
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "p99_abs": float(percentile(diff, 0.99).item()),
        "cosine": float(cosine.item()),
        "p99_rel_floor1": float(percentile(rel, 0.99).item()),
    }


def run_one(ops, name: str, shape: tuple[int, int, int, int], args) -> Result:
    M, K, H, N = shape
    x, gu_w, dn_w, x_s, gu_s, hid_s, dn_s, gate_up, hidden, out = make_case(M, K, H, N)

    def flashrt_call():
        return ops.fp8_swiglu_mlp_bf16(
            x, gu_w, dn_w, x_s, gu_s, hid_s, dn_s, gate_up, hidden, out
        )

    eager_out = torch_ref(x, gu_w, dn_w, x_s, gu_s, hid_s, dn_s)
    flash_out = flashrt_call()
    staged_gate_up = ops.fp8_gemm_bf16(x, gu_w, x_s, gu_s)
    staged_hidden = ops.silu_mul_merged_quantize_fp8_static_bf16(staged_gate_up, hid_s)
    staged_out = ops.fp8_gemm_bf16(staged_hidden, dn_w, hid_s, dn_s)
    torch.cuda.synchronize()
    m = metrics(flash_out, staged_out)
    torch_m = metrics(flash_out, eager_out)
    status = "PASS" if m["p99_abs"] <= args.p99_abs_limit and m["p99_rel_floor1"] <= args.p99_rel_limit else "FAIL"

    flashrt_us = time_us(flashrt_call, args.warmup, args.iters)
    eager_us = time_us(lambda: torch_ref(x, gu_w, dn_w, x_s, gu_s, hid_s, dn_s), args.warmup, args.iters)

    compile_us = None
    compile_status = "skipped"
    if args.compile_baseline:
        try:
            compiled_gemm = torch.compile(ref_fp8_gemm, fullgraph=True, mode="reduce-overhead")

            def compiled_ref():
                gate_up_ref = compiled_gemm(x, gu_w, x_s, gu_s)
                hidden_ref = stable_swiglu_quant_boundary(gate_up_ref, hid_s)
                return compiled_gemm(hidden_ref, dn_w, hid_s, dn_s)

            compiled_out = compiled_ref()
            torch.cuda.synchronize()
            cm = metrics(compiled_out, eager_out)
            if cm["p99_abs"] <= args.p99_abs_limit and cm["p99_rel_floor1"] <= args.p99_rel_limit:
                compile_us = time_us(
                    compiled_ref,
                    args.warmup,
                    args.iters,
                )
                compile_status = "segmented-ok"
            else:
                compile_status = (
                    f"mismatch p99_abs={cm['p99_abs']:.6f} "
                    f"p99_rel={cm['p99_rel_floor1']:.6f}"
                )
        except Exception as exc:
            compile_status = f"failed: {type(exc).__name__}: {exc}"

    return Result(
        shape=name,
        M=M,
        K=K,
        H=H,
        N=N,
        flashrt_us=flashrt_us,
        torch_eager_us=eager_us,
        torch_compile_us=compile_us,
        speedup_vs_eager=eager_us / flashrt_us,
        speedup_vs_compile=(compile_us / flashrt_us if compile_us else None),
        compile_status=compile_status,
        status=status,
        torch_ref_max_abs=torch_m["max_abs"],
        torch_ref_mean_abs=torch_m["mean_abs"],
        torch_ref_p99_abs=torch_m["p99_abs"],
        torch_ref_cosine=torch_m["cosine"],
        torch_ref_p99_rel_floor1=torch_m["p99_rel_floor1"],
        **m,
    )


def write_markdown(path: Path, results: list[Result]) -> None:
    lines = [
        "| Shape | M,K,H,N | FlashRT us | Eager us | vs eager | Compile us | vs compile | Compile status | Staged p99 | Staged cosine | Torch-ref p99 | Torch-ref cosine | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for r in results:
        compile_us = "n/a" if r.torch_compile_us is None else f"{r.torch_compile_us:.3f}"
        speedup_compile = "n/a" if r.speedup_vs_compile is None else f"{r.speedup_vs_compile:.2f}x"
        lines.append(
            f"| {r.shape} | {r.M},{r.K},{r.H},{r.N} | {r.flashrt_us:.3f} | "
            f"{r.torch_eager_us:.3f} | {r.speedup_vs_eager:.2f}x | {compile_us} | "
            f"{speedup_compile} | {r.compile_status} | {r.p99_abs:.6f} | "
            f"{r.cosine:.8f} | {r.torch_ref_p99_abs:.6f} | "
            f"{r.torch_ref_cosine:.8f} | {r.status} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed", "hub"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--repo-id", default="flashrt/flashrt-fp8-swiglu-ffn")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--shapes", choices=sorted(SHAPE_GROUPS), default="smoke")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--compile-baseline", action="store_true")
    parser.add_argument("--p99-abs-limit", type=float, default=1.0)
    parser.add_argument("--p99-rel-limit", type=float, default=0.05)
    parser.add_argument("--output", default=None)
    parser.add_argument("--markdown", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(17)
    ops = {
        "source": load_source_ops,
        "installed": lambda: load_installed_ops(args.artifact),
        "hub": lambda: load_hub_ops(args.repo_id, args.version),
    }[args.backend]()

    results = [run_one(ops, name, SHAPES[name], args) for name in SHAPE_GROUPS[args.shapes]]
    for r in results:
        print(
            f"{r.status} {r.shape}: flashrt={r.flashrt_us:.3f}us "
            f"eager={r.torch_eager_us:.3f}us speedup={r.speedup_vs_eager:.2f}x "
            f"staged_p99={r.p99_abs:.6f} staged_cosine={r.cosine:.8f} "
            f"torch_ref_p99={r.torch_ref_p99_abs:.6f} "
            f"torch_ref_cosine={r.torch_ref_cosine:.8f}"
        )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps([asdict(r) for r in results], indent=2) + "\n")
    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        write_markdown(Path(args.markdown), results)

    if any(r.status != "PASS" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
