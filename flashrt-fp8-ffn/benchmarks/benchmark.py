#!/usr/bin/env python3
"""Benchmark flashrt-fp8-ffn against PyTorch eager/compile references."""

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


SHAPES = {
    # PI0.5 decoder chunks. Production default is 10 denoising steps.
    "pi05_decoder_ffn_m1": (1, 1024, 4096, 1024, 18),
    "pi05_decoder_ffn_m8": (8, 1024, 4096, 1024, 18),
    "pi05_decoder_ffn_m10": (10, 1024, 4096, 1024, 18),
    "pi05_decoder_ffn_m16": (16, 1024, 4096, 1024, 18),
    # Backward-compatible headline alias.
    "pi05_decoder_ffn": (10, 1024, 4096, 1024, 18),
    # PI0.5 SigLIP-L FFN. One view is 256 visual tokens.
    "pi05_vision_ffn_1view": (256, 1152, 4304, 1152, 27),
    "pi05_vision_ffn_2view": (512, 1152, 4304, 1152, 27),
    "pi05_vision_ffn_3view": (768, 1152, 4304, 1152, 27),
    # GROOT/Qwen3-VL ViT FFN.
    "groot_vit_ffn_1view": (256, 1024, 4096, 1024, 24),
    "groot_vit_ffn_2view": (512, 1024, 4096, 1024, 24),
    "groot_vit_ffn_4view": (1024, 1024, 4096, 1024, 24),
    # GROOT DeepStack merger. Two-view ViT taps produce 128 merged tokens.
    "groot_deepstack_merge_2view": (128, 4096, 4096, 2048, 3),
    # GROOT VL self-attention FFN. Sequence length changes with vision/text mix.
    "groot_vl_self_attn_ffn_seq512": (512, 2048, 8192, 2048, 4),
    "groot_vl_self_attn_ffn_seq1024": (1024, 2048, 8192, 2048, 4),
    "groot_vl_self_attn_ffn_seq2520": (2520, 2048, 8192, 2048, 4),
    # Backward-compatible headline alias.
    "groot_vl_self_attn_ffn": (1024, 2048, 8192, 2048, 4),
    # GROOT action DiT GELU FFN. This is exact GELU shape, but the production
    # path currently uses BF16 GEMMs; report it as a shape fit, not a deployed
    # FP8 action-head claim until model wiring is done.
    "groot_action_dit_ffn": (41, 1536, 6144, 1536, 32),
}

SHAPE_GROUPS = {
    "headline": [
        "pi05_decoder_ffn_m10",
        "pi05_vision_ffn_2view",
        "groot_vit_ffn_2view",
        "groot_vl_self_attn_ffn_seq1024",
    ],
    "pi05": [
        "pi05_decoder_ffn_m1",
        "pi05_decoder_ffn_m8",
        "pi05_decoder_ffn_m10",
        "pi05_decoder_ffn_m16",
        "pi05_vision_ffn_1view",
        "pi05_vision_ffn_2view",
        "pi05_vision_ffn_3view",
    ],
    "groot": [
        "groot_vit_ffn_1view",
        "groot_vit_ffn_2view",
        "groot_vit_ffn_4view",
        "groot_deepstack_merge_2view",
        "groot_vl_self_attn_ffn_seq512",
        "groot_vl_self_attn_ffn_seq1024",
        "groot_vl_self_attn_ffn_seq2520",
        "groot_action_dit_ffn",
    ],
}
SHAPE_GROUPS["all"] = SHAPE_GROUPS["pi05"] + SHAPE_GROUPS["groot"]


@dataclass
class Result:
    shape: str
    M: int
    K: int
    H: int
    N: int
    layers: int
    flashrt_us: float
    torch_eager_us: float
    torch_compile_us: float | None
    speedup_vs_eager: float
    speedup_vs_compile: float | None
    compile_status: str
    max_abs: float
    p99_abs: float
    p99_rel_floor1: float
    max_rel_floor1: float
    status: str


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self._ops = getattr(torch.ops, namespace)

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
            hidden_fp8 = torch.empty_like(hidden, dtype=torch.float8_e4m3fn)
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
    namespace = "flashrt_fp8_ffn_benchmark"
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


def load_hub_ops(repo_id: str, version: int):
    from kernels import get_kernel

    return get_kernel(repo_id, version=version)


def quantize_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x.float() / scale.float(), -448.0, 448.0).to(torch.float8_e4m3fn)


def dequant_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x.float() * scale.float()


def compiler_disable(fn):
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and hasattr(compiler, "disable"):
        return compiler.disable(fn)
    return torch._dynamo.disable(fn)


def gelu_quantize_fp8_boundary(
    hidden: torch.Tensor, bias: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    hidden = torch.nn.functional.gelu(
        hidden.float() + bias.float(), approximate="tanh"
    )
    return quantize_fp8(hidden, scale)


def bf16_bias_add_boundary(out: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return (out.float() + bias.float()).to(torch.bfloat16)


stable_gelu_quantize_fp8 = compiler_disable(gelu_quantize_fp8_boundary)
stable_bf16_bias_add = compiler_disable(bf16_bias_add_boundary)


def torch_mlp(x, up_w, up_b, down_w, down_b, x_s, up_s, hid_s, dn_s):
    hidden = (dequant_fp8(x, x_s) @ dequant_fp8(up_w, up_s).T).to(torch.bfloat16)
    hidden = torch.nn.functional.gelu(hidden + up_b.float(), approximate="tanh")
    hidden_fp8 = torch.clamp(hidden / hid_s.float(), -448.0, 448.0).to(torch.float8_e4m3fn)
    out = (dequant_fp8(hidden_fp8, hid_s) @ dequant_fp8(down_w, dn_s).T).to(torch.bfloat16)
    return (out + down_b.float()).to(torch.bfloat16)


def torch_mlp_compile_stable(x, up_w, up_b, down_w, down_b, x_s, up_s, hid_s, dn_s):
    hidden = (dequant_fp8(x, x_s) @ dequant_fp8(up_w, up_s).T).to(torch.bfloat16)
    hidden_fp8 = stable_gelu_quantize_fp8(hidden, up_b, hid_s)
    out = (dequant_fp8(hidden_fp8, hid_s) @ dequant_fp8(down_w, dn_s).T).to(torch.bfloat16)
    return stable_bf16_bias_add(out, down_b)


def make_inputs(M: int, K: int, H: int, N: int, layers: int):
    x_scale = torch.tensor([0.05], device="cuda", dtype=torch.float32)
    up_scale = torch.tensor([0.04], device="cuda", dtype=torch.float32)
    hidden_scale = torch.tensor([0.25], device="cuda", dtype=torch.float32)
    down_scale = torch.tensor([0.04], device="cuda", dtype=torch.float32)
    xs = [
        quantize_fp8(torch.randn((M, K), device="cuda", dtype=torch.bfloat16), x_scale)
        for _ in range(layers)
    ]
    up_ws = [
        quantize_fp8(torch.randn((H, K), device="cuda", dtype=torch.bfloat16), up_scale)
        for _ in range(layers)
    ]
    down_ws = [
        quantize_fp8(torch.randn((N, H), device="cuda", dtype=torch.bfloat16), down_scale)
        for _ in range(layers)
    ]
    up_bs = [torch.randn((H,), device="cuda", dtype=torch.bfloat16) for _ in range(layers)]
    down_bs = [torch.randn((N,), device="cuda", dtype=torch.bfloat16) for _ in range(layers)]
    hidden = [torch.empty((M, H), device="cuda", dtype=torch.bfloat16) for _ in range(layers)]
    hidden_fp8 = [torch.empty((M, H), device="cuda", dtype=torch.float8_e4m3fn) for _ in range(layers)]
    outs = [torch.empty((M, N), device="cuda", dtype=torch.bfloat16) for _ in range(layers)]
    return xs, up_ws, up_bs, down_ws, down_bs, x_scale, up_scale, hidden_scale, down_scale, hidden, hidden_fp8, outs


def time_us(fn, *, warmup: int, iters: int) -> float:
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


def _outputs_close(got, expected) -> bool:
    if isinstance(got, (tuple, list)) and isinstance(expected, (tuple, list)):
        return len(got) == len(expected) and all(
            _outputs_close(g, e) for g, e in zip(got, expected)
        )
    return bool(torch.allclose(got, expected, rtol=3e-2, atol=1.25e-1))


def compile_time_us(fn, expected, *, warmup: int, iters: int) -> tuple[float | None, str]:
    try:
        compiled = torch.compile(fn, fullgraph=False, mode="reduce-overhead")
        compiled_out = compiled()
        torch.cuda.synchronize()
        if not _outputs_close(compiled_out, expected):
            return None, "unsupported: compiled reference output mismatch"
        return time_us(compiled, warmup=warmup, iters=iters), "ok"
    except Exception as exc:  # noqa: BLE001
        return None, f"unsupported: {type(exc).__name__}: {exc}"


def run_shape(ops, name: str, shape, args) -> Result:
    M, K, H, N, layers = shape
    xs, up_ws, up_bs, down_ws, down_bs, x_s, up_s, hid_s, dn_s, hidden, hidden_fp8, outs = make_inputs(
        M, K, H, N, layers
    )

    def flashrt_stack():
        result = []
        for i in range(layers):
            result.append(
                ops.fp8_gelu_mlp_bf16(
                    xs[i],
                    up_ws[i],
                    up_bs[i],
                    down_ws[i],
                    down_bs[i],
                    x_s,
                    up_s,
                    hid_s,
                    dn_s,
                    hidden[i],
                    hidden_fp8[i],
                    outs[i],
                )
            )
        return tuple(result)

    def torch_stack():
        return tuple(
            torch_mlp(xs[i], up_ws[i], up_bs[i], down_ws[i], down_bs[i], x_s, up_s, hid_s, dn_s)
            for i in range(layers)
        )

    def torch_stack_compile_stable():
        return tuple(
            torch_mlp_compile_stable(
                xs[i], up_ws[i], up_bs[i], down_ws[i], down_bs[i], x_s, up_s, hid_s, dn_s
            )
            for i in range(layers)
        )

    flashrt_stack()
    expected0 = torch_mlp(xs[0], up_ws[0], up_bs[0], down_ws[0], down_bs[0], x_s, up_s, hid_s, dn_s)
    diff = (outs[0].float() - expected0.float()).abs().flatten()
    rel = diff / expected0.float().abs().flatten().clamp_min(1.0)
    max_abs = float(diff.max().item())
    p99_abs = float(percentile(diff, 0.99).item())
    p99_rel = float(percentile(rel, 0.99).item())
    max_rel = float(rel.max().item())
    status = (
        "PASS"
        if p99_abs <= args.p99_abs_limit and p99_rel <= args.p99_rel_floor1_limit
        else "FAIL"
    )

    flashrt_us = time_us(flashrt_stack, warmup=args.warmup, iters=args.iters)
    eager_us = time_us(torch_stack, warmup=args.warmup, iters=args.iters)
    compile_us = None
    compile_status = "not_requested"
    if args.compile_baseline:
        eager_expected = torch_stack()
        stable_expected = torch_stack_compile_stable()
        torch.cuda.synchronize()
        if not _outputs_close(stable_expected, eager_expected):
            compile_status = "unsupported: stable compile reference differs from eager"
        else:
            compile_us, compile_status = compile_time_us(
                torch_stack_compile_stable,
                eager_expected,
                warmup=args.warmup,
                iters=args.iters,
            )

    return Result(
        shape=name,
        M=M,
        K=K,
        H=H,
        N=N,
        layers=layers,
        flashrt_us=flashrt_us,
        torch_eager_us=eager_us,
        torch_compile_us=compile_us,
        speedup_vs_eager=eager_us / flashrt_us,
        speedup_vs_compile=compile_us / flashrt_us if compile_us is not None else None,
        compile_status=compile_status,
        max_abs=max_abs,
        p99_abs=p99_abs,
        p99_rel_floor1=p99_rel,
        max_rel_floor1=max_rel,
        status=status,
    )


def write_markdown(path: Path, results: list[Result], args) -> None:
    lines = [
        "# Benchmark Results: flashrt-fp8-ffn",
        "",
        f"- Backend: `{args.backend}`",
        f"- Device: `{torch.cuda.get_device_name(0)}`",
        f"- Torch: `{torch.__version__}`",
        f"- Warmup/iters: `{args.warmup}/{args.iters}`",
        f"- Precision gate: p99_abs <= `{args.p99_abs_limit}` and "
        f"p99_rel_floor1 <= `{args.p99_rel_floor1_limit}`",
        "- Compile baseline: reported only when compiled reference output "
        "matches eager reference output.",
        "",
        "| Shape | M,K,H,N | Layers | FlashRT us | Eager us | vs eager | Compile us | vs compile | Compile status | P99 abs | P99 rel | Max abs | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for r in results:
        compile_us = f"{r.torch_compile_us:.3f}" if r.torch_compile_us is not None else "n/a"
        compile_speedup = f"{r.speedup_vs_compile:.2f}x" if r.speedup_vs_compile is not None else "n/a"
        lines.append(
            f"| {r.shape} | {r.M},{r.K},{r.H},{r.N} | {r.layers} | "
            f"{r.flashrt_us:.3f} | {r.torch_eager_us:.3f} | {r.speedup_vs_eager:.2f}x | "
            f"{compile_us} | {compile_speedup} | {r.compile_status} | {r.p99_abs:.4f} | "
            f"{r.p99_rel_floor1:.6f} | {r.max_abs:.4f} | {r.status} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed", "hub"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--repo-id", default="flashrt/flashrt-fp8-ffn")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--shapes", default="all")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--compile-baseline", action="store_true")
    parser.add_argument("--p99-abs-limit", type=float, default=1.0)
    parser.add_argument("--p99-rel-floor1-limit", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--list-shapes", action="store_true")
    args = parser.parse_args()

    if args.list_shapes:
        print("Shape groups:")
        for group, names in SHAPE_GROUPS.items():
            print(f"  {group}: {','.join(names)}")
        print("\nShapes:")
        for name, shape in SHAPES.items():
            print(f"  {name}: M,K,H,N,layers={shape}")
        return

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(17)
    if args.backend == "source":
        ops = load_source_ops()
    elif args.backend == "installed":
        ops = load_installed_ops(args.artifact)
    else:
        ops = load_hub_ops(args.repo_id, args.version)
    requested = [s.strip() for s in args.shapes.split(",")]
    names: list[str] = []
    for item in requested:
        if item in SHAPE_GROUPS:
            names.extend(SHAPE_GROUPS[item])
        else:
            names.append(item)
    unknown = [name for name in names if name not in SHAPES]
    if unknown:
        raise SystemExit(f"unknown shapes/groups: {unknown}")

    results = []
    for name in names:
        results.append(run_shape(ops, name, SHAPES[name], args))
        torch.cuda.empty_cache()

    for r in results:
        compile_part = (
            f", compile={r.torch_compile_us:.3f}us, vs_compile={r.speedup_vs_compile:.2f}x"
            if r.torch_compile_us is not None
            else f", compile={r.compile_status}"
        )
        print(
            f"{r.shape}: flashrt={r.flashrt_us:.3f}us, eager={r.torch_eager_us:.3f}us, "
            f"vs_eager={r.speedup_vs_eager:.2f}x{compile_part}, "
            f"p99_abs={r.p99_abs:.4f}, max_abs={r.max_abs:.4f}, {r.status}"
        )

    payload = {
        "backend": args.backend,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "results": [asdict(r) for r in results],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.markdown, results, args)


if __name__ == "__main__":
    main()
