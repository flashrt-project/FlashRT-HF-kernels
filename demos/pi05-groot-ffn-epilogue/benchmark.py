#!/usr/bin/env python3
"""PI0.5/GROOT-shaped FFN FP8 epilogue stack benchmark.

This demo measures reusable FFN post-GEMM epilogue blocks that appear repeatedly
in PI0.5 and GROOT-style VLA/VLM backbones:

* bias + GELU(tanh) + static FP8 cast
* GELU(tanh) + static FP8 cast
* per-channel scale + static FP8 cast

It is a model-block benchmark, not a full model generation benchmark.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import importlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "flashrt-gemm-epilogues"
REGISTRATION_INCLUDE = (
    ROOT.parent
    / "kernels"
    / "kernel-builder"
    / "src"
    / "pyproject"
    / "templates"
    / "torch"
)


BLOCKS = {
    # PI0.5 SigLIP FFN: 2 camera views * 256 tokens, hidden 4304, 27 layers.
    "pi05_vision_ffn_2view": {
        "family": "PI0.5 vision SigLIP FFN",
        "op": "bias_gelu",
        "rows": 512,
        "hidden": 4304,
        "layers": 27,
        "note": "SigLIP FFN fc1 epilogue shape: bias + GELU + FP8 cast.",
    },
    # PI0.5 encoder activation quant before FP8 GEMMs. Prompt default is 48.
    "pi05_encoder_channel_scale": {
        "family": "PI0.5 Gemma encoder activation quant",
        "op": "channel_scale",
        "rows": 2 * 256 + 48,
        "hidden": 2048,
        "layers": 18,
        "note": "Encoder-sequence activation scaling/FP8 cast shape.",
    },
    # GROOT N1.7 ViT FFN: 2 views * 256 tokens, hidden 4096, 24 layers.
    "groot_vit_ffn_2view": {
        "family": "GROOT N1.7 ViT FFN",
        "op": "bias_gelu",
        "rows": 512,
        "hidden": 4096,
        "layers": 24,
        "note": "ViT fc1 epilogue shape in the FP8 backbone.",
    },
    # GROOT DeepStack merger: visual tokens reduced by 4, hidden 4096, 3 taps.
    "groot_deepstack_merge": {
        "family": "GROOT N1.7 DeepStack merger",
        "op": "bias_gelu",
        "rows": 128,
        "hidden": 4096,
        "layers": 3,
        "note": "DeepStack merger fc1 epilogue shape for two-view visual input.",
    },
    # GROOT VL self-attn FFN: long VLM sequence, hidden 8192, 4 layers.
    "groot_vl_self_attn_ffn_long": {
        "family": "GROOT N1.7 VL self-attn FFN",
        "op": "bias_gelu",
        "rows": 1024,
        "hidden": 8192,
        "layers": 4,
        "note": "Long VLM sequence fc1 epilogue shape in VL self-attn blocks.",
    },
}


@dataclass
class Result:
    block: str
    family: str
    op: str
    rows: int
    hidden: int
    layers: int
    flashrt_stack_us: float
    flashrt_per_layer_us: float
    torch_eager_stack_us: float
    torch_eager_per_layer_us: float
    torch_compile_stack_us: float | None
    torch_compile_per_layer_us: float | None
    speedup_vs_eager: float
    speedup_vs_compile: float | None
    compile_status: str
    exact_mismatches: int
    max_abs: float
    status: str
    note: str


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self._ops = getattr(torch.ops, namespace)

    def bias_gelu_quantize_fp8_static_bf16(
        self,
        x: torch.Tensor,
        bias: torch.Tensor,
        scale: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if out is None:
            out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        self._ops.bias_gelu_quantize_fp8_static_bf16(x, bias, scale, out)
        return out

    def gelu_quantize_fp8_static_bf16(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if out is None:
            out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        self._ops.gelu_quantize_fp8_static_bf16(x, scale, out)
        return out

    def channel_scale_quantize_fp8_static_bf16(
        self,
        x: torch.Tensor,
        channel_scale: torch.Tensor,
        scale: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if out is None:
            out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        self._ops.channel_scale_quantize_fp8_static_bf16(
            x, channel_scale, scale, out
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
        raise RuntimeError(
            f"missing kernel-builder registration include: {REGISTRATION_INCLUDE}"
        )
    _preload_cublaslt()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _current_arch_list())
    namespace = "flashrt_pi05_groot_ffn_demo"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "bf16_gemm_bias_gelu.cu"),
            str(PACKAGE / "csrc" / "bias_gelu_quantize_fp8.cu"),
            str(PACKAGE / "csrc" / "channel_scale_quantize_fp8.cu"),
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
        return importlib.import_module("flashrt_gemm_epilogues")
    finally:
        if artifact:
            sys.path.remove(artifact)


def load_hub_ops(repo_id: str, version: int):
    from kernels import get_kernel

    return get_kernel(repo_id, version=version, trust_remote_code=True)


def load_ops(args):
    if args.backend == "source":
        return load_source_ops()
    if args.backend == "installed":
        return load_installed_ops(args.artifact)
    return load_hub_ops(args.repo_id, args.version)


def torch_bias_gelu_fp8(
    x: torch.Tensor,
    bias: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    y = x.float() + bias.float()
    y = torch.nn.functional.gelu(y, approximate="tanh")
    y = torch.clamp(y / scale.float(), -448.0, 448.0)
    return y.to(torch.float8_e4m3fn)


def torch_gelu_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    y = torch.nn.functional.gelu(x.float(), approximate="tanh")
    y = torch.clamp(y / scale.float(), -448.0, 448.0)
    return y.to(torch.float8_e4m3fn)


def torch_channel_scale_fp8(
    x: torch.Tensor,
    channel_scale: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    y = x.float() * channel_scale.float()
    y = torch.clamp(y / scale.float(), -448.0, 448.0)
    return y.to(torch.float8_e4m3fn)


def _time_us(fn: Callable[[], object], *, warmup: int, iters: int) -> float:
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


def _measure_compile(fn: Callable[[], object], *, warmup: int, iters: int) -> tuple[float | None, str]:
    try:
        compiled = torch.compile(fn, fullgraph=True, mode="reduce-overhead")
        compiled()
        torch.cuda.synchronize()
        return _time_us(compiled, warmup=warmup, iters=iters), "ok"
    except Exception as exc:  # noqa: BLE001 - report benchmark backend failure.
        return None, f"unsupported: {type(exc).__name__}: {exc}"


def _make_inputs(rows: int, hidden: int, layers: int):
    xs = [
        torch.randn((rows, hidden), device="cuda", dtype=torch.bfloat16)
        for _ in range(layers)
    ]
    biases = [
        torch.randn((hidden,), device="cuda", dtype=torch.bfloat16)
        for _ in range(layers)
    ]
    channel_scales = [
        torch.randn((hidden,), device="cuda", dtype=torch.bfloat16)
        for _ in range(layers)
    ]
    outs = [
        torch.empty((rows, hidden), device="cuda", dtype=torch.float8_e4m3fn)
        for _ in range(layers)
    ]
    scale = torch.tensor([0.25], device="cuda", dtype=torch.float32)
    return xs, biases, channel_scales, outs, scale


def run_block(
    ops,
    name: str,
    cfg: dict,
    *,
    warmup: int,
    iters: int,
    compile_baseline: bool,
) -> Result:
    rows = int(cfg["rows"])
    hidden = int(cfg["hidden"])
    layers = int(cfg["layers"])
    op_name = str(cfg["op"])
    xs, biases, channel_scales, outs, scale = _make_inputs(rows, hidden, layers)

    def flashrt_stack() -> tuple[torch.Tensor, ...]:
        for i in range(layers):
            if op_name == "bias_gelu":
                ops.bias_gelu_quantize_fp8_static_bf16(
                    xs[i], biases[i], scale, out=outs[i]
                )
            elif op_name == "gelu":
                ops.gelu_quantize_fp8_static_bf16(xs[i], scale, out=outs[i])
            elif op_name == "channel_scale":
                ops.channel_scale_quantize_fp8_static_bf16(
                    xs[i], channel_scales[i], scale, out=outs[i]
                )
            else:
                raise ValueError(f"unknown op: {op_name}")
        return tuple(outs)

    def torch_stack() -> tuple[torch.Tensor, ...]:
        results = []
        for i in range(layers):
            if op_name == "bias_gelu":
                results.append(torch_bias_gelu_fp8(xs[i], biases[i], scale))
            elif op_name == "gelu":
                results.append(torch_gelu_fp8(xs[i], scale))
            elif op_name == "channel_scale":
                results.append(torch_channel_scale_fp8(xs[i], channel_scales[i], scale))
            else:
                raise ValueError(f"unknown op: {op_name}")
        return tuple(results)

    flashrt_stack()
    torch.cuda.synchronize()
    if op_name == "bias_gelu":
        expected = torch_bias_gelu_fp8(xs[0], biases[0], scale)
    elif op_name == "gelu":
        expected = torch_gelu_fp8(xs[0], scale)
    else:
        expected = torch_channel_scale_fp8(xs[0], channel_scales[0], scale)
    got = outs[0]
    mismatches = int((got.detach().cpu() != expected.detach().cpu()).sum().item())
    max_abs = float((got.float() - expected.float()).abs().max().item())

    flashrt_us = _time_us(flashrt_stack, warmup=warmup, iters=iters)
    torch_us = _time_us(torch_stack, warmup=warmup, iters=iters)
    compile_us = None
    compile_status = "not_requested"
    if compile_baseline:
        compile_us, compile_status = _measure_compile(
            torch_stack, warmup=warmup, iters=iters
        )

    return Result(
        block=name,
        family=str(cfg["family"]),
        op=op_name,
        rows=rows,
        hidden=hidden,
        layers=layers,
        flashrt_stack_us=flashrt_us,
        flashrt_per_layer_us=flashrt_us / layers,
        torch_eager_stack_us=torch_us,
        torch_eager_per_layer_us=torch_us / layers,
        torch_compile_stack_us=compile_us,
        torch_compile_per_layer_us=compile_us / layers if compile_us is not None else None,
        speedup_vs_eager=torch_us / flashrt_us,
        speedup_vs_compile=compile_us / flashrt_us if compile_us is not None else None,
        compile_status=compile_status,
        exact_mismatches=mismatches,
        max_abs=max_abs,
        status="PASS" if mismatches == 0 else "FAIL",
        note=str(cfg["note"]),
    )


def write_markdown(path: Path, results: list[Result], args) -> None:
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA"
    lines = [
        "# PI0.5/GROOT FFN Epilogue Demo Results",
        "",
        f"- Backend: `{args.backend}`",
        f"- Device: `{device}`",
        f"- Torch: `{torch.__version__}`",
        f"- Warmup/iters: `{args.warmup}/{args.iters}`",
        "",
        "This is a model-block benchmark for repeated FFN epilogue/activation-quant blocks, not a full model generation benchmark.",
        "",
        "| Block | Op | Shape | Layers | FlashRT stack us | Torch eager stack us | Speedup eager | Torch compile stack us | Speedup compile | Exact | Note |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        compile_us = f"{r.torch_compile_stack_us:.3f}" if r.torch_compile_stack_us is not None else "n/a"
        compile_speedup = f"{r.speedup_vs_compile:.2f}x" if r.speedup_vs_compile is not None else "n/a"
        lines.append(
            f"| {r.block} | {r.op} | {r.rows}x{r.hidden} | {r.layers} | "
            f"{r.flashrt_stack_us:.3f} | {r.torch_eager_stack_us:.3f} | "
            f"{r.speedup_vs_eager:.2f}x | {compile_us} | {compile_speedup} | "
            f"{r.status} ({r.exact_mismatches}) | {r.note} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_blocks(value: str) -> list[str]:
    if value == "all":
        return list(BLOCKS)
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in names if item not in BLOCKS]
    if unknown:
        raise ValueError(f"unknown blocks: {unknown}; available={list(BLOCKS)}")
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed", "hub"], default="source")
    parser.add_argument("--artifact", default=None, help="Path added to sys.path for installed backend.")
    parser.add_argument("--repo-id", default="flashrt/flashrt-gemm-epilogues")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--blocks", default="all", help="'all' or comma-separated block names.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--compile-baseline", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    torch.manual_seed(7)
    ops = load_ops(args)
    block_names = parse_blocks(args.blocks)
    results = [
        run_block(
            ops,
            name,
            BLOCKS[name],
            warmup=args.warmup,
            iters=args.iters,
            compile_baseline=args.compile_baseline,
        )
        for name in block_names
    ]

    for r in results:
        compile_part = (
            f", compile={r.torch_compile_stack_us:.3f}us, "
            f"vs_compile={r.speedup_vs_compile:.2f}x"
            if r.torch_compile_stack_us is not None
            else f", compile={r.compile_status}"
        )
        print(
            f"{r.block}: flashrt={r.flashrt_stack_us:.3f}us, "
            f"eager={r.torch_eager_stack_us:.3f}us, "
            f"vs_eager={r.speedup_vs_eager:.2f}x{compile_part}, "
            f"exact={r.status}"
        )

    payload = {
        "backend": args.backend,
        "repo_id": args.repo_id if args.backend == "hub" else "",
        "version": args.version if args.backend == "hub" else "",
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "results": [asdict(r) for r in results],
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.markdown, results, args)


if __name__ == "__main__":
    main()
