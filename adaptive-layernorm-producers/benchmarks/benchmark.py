#!/usr/bin/env python3
"""Benchmark adaptive-layernorm-producers against eager producer chains."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "adaptive-layernorm-producers" / "tests"))
from test_adaptive_layernorm_producers import (  # noqa: E402
    _nvfp4_out,
    load_source_ops,
    make_case,
    quant_fp8,
    ref_adaln,
    ref_layer_norm_no_affine,
)


def load_installed_ops(artifact: str | None):
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return importlib.import_module("adaptive_layernorm_producers")
    finally:
        if artifact:
            sys.path.remove(artifact)


def time_cuda(fn, iters: int = 200, warmup: int = 50) -> float:
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


def graph_time_cuda(
    fn, iters: int = 200, warmup: int = 50, launches_per_graph: int = 32
) -> float:
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(launches_per_graph):
            fn()
    return (
        time_cuda(graph.replay, iters=iters, warmup=warmup)
        / launches_per_graph
    )


def load_native():
    from torch.utils.cpp_extension import load

    major, minor = torch.cuda.get_device_capability()
    os.environ.setdefault(
        "TORCH_CUDA_ARCH_LIST", "12.0a" if major >= 12 else f"{major}.{minor}"
    )
    package = ROOT / "adaptive-layernorm-producers"
    return load(
        name="adaptive_layernorm_producers_native_bench",
        sources=[
            str(package / "benchmarks" / "native_binding.cpp"),
            str(package / "csrc" / "adaln_modulation6.cu"),
        ],
        extra_include_paths=[str(package / "csrc")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
        verbose=False,
    )


def run_modulation6(ops, native, batch, sequence, dim, iters):
    params = torch.randn(
        (batch, sequence, 6, dim), device="cuda", dtype=torch.float32
    )
    modulation = torch.randn((6, dim), device="cuda", dtype=torch.float32)
    native_out = tuple(
        torch.empty(
            (batch, sequence, dim), device="cuda", dtype=torch.bfloat16
        )
        for _ in range(6)
    )
    wrapper_out = tuple(torch.empty_like(native_out[0]) for _ in range(6))

    def native_call():
        native.adaln_modulation6(
            params.data_ptr(), modulation.data_ptr(),
            *(value.data_ptr() for value in native_out),
            batch, sequence, dim,
        )

    def wrapper_call():
        return ops.adaln_modulation6_bf16(
            params, modulation, out=wrapper_out
        )

    def eager_call():
        return tuple(
            (params[:, :, index, :] + modulation[index]).bfloat16()
            for index in range(6)
        )

    compiled_call = torch.compile(eager_call, fullgraph=True)
    native_call()
    wrapper_call()
    for raw, wrapped in zip(native_out, wrapper_out, strict=True):
        torch.testing.assert_close(raw, wrapped, rtol=0, atol=0)
    native_us = time_cuda(native_call, iters=iters)
    wrapper_us = time_cuda(wrapper_call, iters=iters)
    eager_us = time_cuda(eager_call, iters=iters)
    compile_us = time_cuda(compiled_call, iters=iters)
    graph_native_us = graph_time_cuda(native_call, iters=iters)
    graph_wrapper_us = graph_time_cuda(wrapper_call, iters=iters)
    return (
        native_us, wrapper_us, graph_native_us, graph_wrapper_us,
        eager_us, compile_us
    )


def run_case(ops, name: str, rows: int, dim: int, eps: float, iters: int) -> dict[str, float | str | int]:
    x, scale, shift, _inv_s, act_scale, _scale_fp8, _shift_fp8, _scale_deq, _shift_deq = make_case(rows, dim)
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)

    def fused():
        ops.ada_layer_norm_quant_fp8_bf16(x, scale, shift, act_scale, eps, out=out)

    def eager():
        quant_fp8(ref_adaln(x, scale, shift, eps), act_scale)

    no_affine_out = torch.empty_like(x, dtype=torch.float8_e4m3fn)

    def fused_no_affine():
        ops.layer_norm_no_affine_quant_fp8_static_bf16(x, act_scale, eps, out=no_affine_out)

    def eager_no_affine():
        quant_fp8(ref_layer_norm_no_affine(x, eps), act_scale)

    fused_us = time_cuda(fused, iters=iters)
    eager_us = time_cuda(eager, iters=iters)
    fused_no_affine_us = time_cuda(fused_no_affine, iters=iters)
    eager_no_affine_us = time_cuda(eager_no_affine, iters=iters)
    return {
        "shape": name,
        "rows": rows,
        "dim": dim,
        "ada_fp8_us": fused_us,
        "ada_eager_us": eager_us,
        "ada_speedup": eager_us / fused_us,
        "no_affine_fp8_us": fused_no_affine_us,
        "no_affine_eager_us": eager_no_affine_us,
        "no_affine_speedup": eager_no_affine_us / fused_no_affine_us,
    }


def run_ptok_table(ops, name: str, rows: int, dim: int, eps: float, iters: int):
    chunks = 6
    x = torch.randn((rows, dim), device="cuda", dtype=torch.bfloat16)
    temb = torch.randn(
        (rows, chunks, dim), device="cuda", dtype=torch.bfloat16
    )
    table = torch.randn((chunks, dim), device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    packed, sf = _nvfp4_out(x)

    def fused_bf16():
        return ops.ada_layer_norm_ptok_table_bf16(
            x, temb, table, 0, 1, eps, out=out
        )

    def fused_nvfp4():
        return ops.ada_layer_norm_quant_nvfp4_swizzled_ptok_table_bf16(
            x, temb, table, 0, 1, eps, packed=packed, sf_swizzled=sf
        )

    def eager_bf16():
        xf = x.float()
        mean = xf.mean(dim=-1, keepdim=True)
        var = (xf - mean).square().mean(dim=-1, keepdim=True)
        norm = (xf - mean) * torch.rsqrt(var + eps)
        shift = temb[:, 0].float() + table[0]
        scale = temb[:, 1].float() + table[1]
        return (norm * (1.0 + scale) + shift).to(torch.bfloat16)

    return {
        "shape": name,
        "rows": rows,
        "dim": dim,
        "bf16_us": time_cuda(fused_bf16, iters=iters),
        "eager_us": time_cuda(eager_bf16, iters=iters),
        "nvfp4_us": time_cuda(fused_nvfp4, iters=iters),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--markdown", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(2026)
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    native = load_native()
    shapes = [
        ("decode_action", 16, 2048),
        ("wan_video_short", 64, 3072),
        ("wan_video_ctx", 256, 3072),
        ("wan_video_2k", 2520, 3072),
        ("wan_video_4k", 4096, 3072),
    ]
    rows = [run_case(ops, name, r, d, 1e-5, args.iters) for name, r, d in shapes]
    table_rows = [
        run_ptok_table(ops, name, r, d, 1e-5, args.iters)
        for name, r, d in [
            ("video_short", 64, 3072),
            ("video_long", 2520, 3072),
        ]
    ]
    modulation_rows = []
    for name, batch, sequence, dim in [
        ("groot_dit", 1, 51, 1536),
        ("motus", 1, 2520, 3072),
        ("video_long", 1, 5070, 3072),
    ]:
        modulation_rows.append(
            (name, batch, sequence, dim)
            + run_modulation6(ops, native, batch, sequence, dim, args.iters)
        )
    lines = [
        "| Shape | Rows | Dim | AdaLN->FP8 us | Eager chain us | Speedup | LN->FP8 us | Eager LN chain us | Speedup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        line = (
            f"| {row['shape']} | {row['rows']} | {row['dim']} | "
            f"{row['ada_fp8_us']:.3f} | {row['ada_eager_us']:.3f} | {row['ada_speedup']:.2f}x | "
            f"{row['no_affine_fp8_us']:.3f} | {row['no_affine_eager_us']:.3f} | {row['no_affine_speedup']:.2f}x |"
        )
        lines.append(line)
        print(line)
    lines.extend([
        "",
        "| Per-token table shape | BF16 fused us | BF16 eager us | Speedup | NVFP4 fused us |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in table_rows:
        line = (
            f"| {row['shape']} M{row['rows']} D{row['dim']} | "
            f"{row['bf16_us']:.3f} | {row['eager_us']:.3f} | "
            f"{row['eager_us'] / row['bf16_us']:.2f}x | {row['nvfp4_us']:.3f} |"
        )
        lines.append(line)
        print(line)
    lines.extend(
        [
            "",
            "| Modulation6 shape | Native us | Wrapper us | Graph native us | Graph wrapper us | Eager us | Compile us | Wrapper/native | Graph wrapper/native |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for (
        name, batch, sequence, dim, native_us, wrapper_us, graph_native_us,
        graph_wrapper_us, eager_us, compile_us
    ) in modulation_rows:
        line = (
            f"| {name} B{batch} S{sequence} D{dim} | {native_us:.3f} | "
            f"{wrapper_us:.3f} | {graph_native_us:.3f} | "
            f"{graph_wrapper_us:.3f} | {eager_us:.3f} | {compile_us:.3f} | "
            f"{wrapper_us / native_us:.3f} | "
            f"{graph_wrapper_us / graph_native_us:.3f} |"
        )
        lines.append(line)
        print(line)
    if args.markdown:
        Path(args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
