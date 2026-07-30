#!/usr/bin/env python3
"""Benchmark Tensor wrappers against native FlashRT and torch baselines."""

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
PACKAGE = ROOT / "flashrt-spatiotemporal-layout"
REGISTRATION = (
    ROOT.parent / "kernels/kernel-builder/src/pyproject/templates/torch"
)
SHAPES = {
    "latent-small": (1, 64, 4, 32, 32),
    "vae-channel320": (1, 320, 17, 32, 32),
    "vae-channel512": (1, 512, 4, 64, 64),
}


@dataclass
class Result:
    workload: str
    shape: str
    dtype: str
    native_us: float
    wrapper_us: float
    native_parity: float
    graph_native_us: float
    graph_wrapper_us: float
    graph_native_parity: float
    eager_us: float
    compile_us: float
    strong_library_us: str
    max_abs: float
    mean_abs: float
    p99_abs: float
    cosine: float
    accepted: bool


def time_us(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return begin.elapsed_time(end) * 1000.0 / iters


def graph_time_us(fn, warmup: int, iters: int) -> float:
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return time_us(graph.replay, warmup, iters)


def build_source():
    from torch.utils.cpp_extension import load

    major, minor = torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    namespace = "flashrt_spatiotemporal_layout_native_parity"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext/torch_binding.cpp"),
            str(PACKAGE / "csrc/spatiotemporal_layout.cu"),
            str(PACKAGE / "csrc/bf16_ndhwc_to_ncdhw_transpose.cu"),
            str(PACKAGE / "csrc/bf16_quant_fp8_ncdhw_to_ndhwc.cu"),
        ],
        extra_include_paths=[str(PACKAGE / "csrc"), str(REGISTRATION)],
        extra_cflags=["-O3", "-DCUDA_KERNEL"],
        extra_cuda_cflags=["-O3", "-DCUDA_KERNEL"],
        is_python_module=False,
        verbose=False,
    )
    return getattr(torch.ops, namespace)


def load_wrapper(backend: str, artifact: str | None):
    if backend == "source":
        return build_source()
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return importlib.import_module("flashrt_spatiotemporal_layout")
    finally:
        if artifact:
            sys.path.remove(artifact)


def build_native():
    from torch.utils.cpp_extension import load

    return load(
        name="flashrt_spatiotemporal_layout_raw_native",
        sources=[
            str(PACKAGE / "benchmarks/native_binding.cpp"),
            str(PACKAGE / "csrc/spatiotemporal_layout.cu"),
            str(PACKAGE / "csrc/bf16_ndhwc_to_ncdhw_transpose.cu"),
            str(PACKAGE / "csrc/bf16_quant_fp8_ncdhw_to_ndhwc.cu"),
        ],
        extra_include_paths=[str(PACKAGE / "csrc")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


def metrics(got: torch.Tensor, ref: torch.Tensor):
    if got.dtype == torch.float8_e4m3fn:
        exact = torch.equal(got.view(torch.uint8), ref.view(torch.uint8))
        return (0.0, 0.0, 0.0, 1.0) if exact else (float("inf"),) * 4
    diff = (got.float() - ref.float()).abs().flatten()
    cosine = torch.nn.functional.cosine_similarity(
        got.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    return (
        diff.max().item(),
        diff.mean().item(),
        torch.quantile(diff, 0.99).item(),
        cosine,
    )


def add_result(
    rows,
    workload,
    shape,
    wrapper_fn,
    native_fn,
    eager_fn,
    compiled_fn,
    got,
    ref,
    args,
):
    native_us = time_us(native_fn, args.warmup, args.iters)
    wrapper_us = time_us(wrapper_fn, args.warmup, args.iters)
    graph_native_us = graph_time_us(native_fn, args.warmup, args.iters)
    graph_wrapper_us = graph_time_us(wrapper_fn, args.warmup, args.iters)
    eager_us = time_us(eager_fn, args.warmup, args.iters)
    compile_us = time_us(compiled_fn, args.warmup, args.iters)
    max_abs, mean_abs, p99_abs, cosine = metrics(got, ref)
    parity = wrapper_us / native_us
    rows.append(
        Result(
            workload,
            str(shape),
            str(got.dtype),
            native_us,
            wrapper_us,
            parity,
            graph_native_us,
            graph_wrapper_us,
            graph_wrapper_us / graph_native_us,
            eager_us,
            compile_us,
            "N/A (no equivalent single library op)",
            max_abs,
            mean_abs,
            p99_abs,
            cosine,
            (
                wrapper_us - native_us <= max(0.75, native_us * 0.05)
                or graph_wrapper_us - graph_native_us
                <= max(0.5, graph_native_us * 0.05)
            )
            and min(wrapper_us, graph_wrapper_us)
            <= min(eager_us, compile_us) * 0.98
            and max_abs == 0.0,
        )
    )


def run_shape(wrapper, native, name, shape, args):
    b, c, t, h, w = shape
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    ndhwc = x.permute(0, 2, 3, 4, 1).contiguous()
    bias = torch.randn(c, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    scale = 0.03125
    rows = []

    def wrapper_layout():
        wrapper.ndhwc_to_ncdhw_bf16(ndhwc, out_layout)

    def eager_layout():
        out_layout.copy_(ndhwc.permute(0, 4, 1, 2, 3))

    compiled_layout = torch.compile(eager_layout, fullgraph=True)
    out_layout = torch.empty_like(x)
    native_layout = lambda: native.ndhwc_to_ncdhw(ndhwc, out_layout)
    wrapper_layout()
    ref = ndhwc.permute(0, 4, 1, 2, 3).contiguous()
    add_result(
        rows, f"{name}/ndhwc_to_ncdhw", shape, wrapper_layout, native_layout,
        eager_layout, compiled_layout, out_layout, ref, args
    )

    out_bias = torch.empty_like(x)
    wrapper_bias = lambda: wrapper.ndhwc_to_ncdhw_bias_bf16(
        ndhwc, bias, out_bias
    )
    native_bias = lambda: native.ndhwc_to_ncdhw_bias(ndhwc, bias, out_bias)

    def eager_bias():
        out_bias.copy_(
            (ndhwc.permute(0, 4, 1, 2, 3).float()
             + bias.float().view(1, c, 1, 1, 1)).to(torch.bfloat16)
        )

    compiled_bias = torch.compile(eager_bias, fullgraph=True)
    wrapper_bias()
    ref_bias = (
        x.float() + bias.float().view(1, c, 1, 1, 1)
    ).to(torch.bfloat16)
    add_result(
        rows, f"{name}/ndhwc_to_ncdhw_bias", shape, wrapper_bias, native_bias,
        eager_bias, compiled_bias, out_bias, ref_bias, args
    )

    out_add = torch.empty_like(x)
    wrapper_add = lambda: wrapper.ndhwc_to_ncdhw_add_bf16(
        ndhwc, residual, out_add
    )
    native_add = lambda: native.ndhwc_to_ncdhw_add(
        ndhwc, residual, out_add
    )

    def eager_add():
        out_add.copy_(
            (ndhwc.permute(0, 4, 1, 2, 3).float()
             + residual.float()).to(torch.bfloat16)
        )

    compiled_add = torch.compile(eager_add, fullgraph=True)
    wrapper_add()
    ref_add = (x.float() + residual.float()).to(torch.bfloat16)
    add_result(
        rows, f"{name}/ndhwc_to_ncdhw_add", shape, wrapper_add, native_add,
        eager_add, compiled_add, out_add, ref_add, args
    )

    out_fp8 = torch.empty(
        (b, t, h, w, c), device="cuda", dtype=torch.float8_e4m3fn
    )
    wrapper_quant = lambda: wrapper.ncdhw_quantize_fp8_static_ndhwc_bf16(
        x, scale, out_fp8
    )
    native_quant = lambda: native.ncdhw_quantize(x, scale, out_fp8)

    def eager_quant():
        out_fp8.copy_(
            (x.float() / scale).clamp(-448.0, 448.0)
            .to(torch.float8_e4m3fn).permute(0, 2, 3, 4, 1)
        )

    compiled_quant = torch.compile(eager_quant, fullgraph=True)
    wrapper_quant()
    ref_fp8 = (
        (x.float() / scale).clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn).permute(0, 2, 3, 4, 1).contiguous()
    )
    add_result(
        rows, f"{name}/ncdhw_quantize_fp8_ndhwc", shape, wrapper_quant,
        native_quant, eager_quant, compiled_quant, out_fp8, ref_fp8, args
    )
    return rows


def run_cache_and_upsample(wrapper, native, args):
    rows = []
    previous = torch.randn(
        (1, 64, 2, 32, 32), device="cuda", dtype=torch.bfloat16
    )
    current = torch.randn(
        (1, 64, 1, 32, 32), device="cuda", dtype=torch.bfloat16
    )
    packed = torch.empty(
        (1, 32, 32, 192), device="cuda", dtype=torch.bfloat16
    )
    wrapper_pack = lambda: wrapper.pack_causal_cache3_nhwc_bf16(
        previous, current, packed
    )
    native_pack = lambda: native.pack_causal_cache3_nhwc(
        previous, current, packed
    )

    def eager_pack():
        packed.copy_(
            torch.cat(
                (previous[:, :, 0], previous[:, :, 1], current[:, :, 0]),
                dim=1,
            ).permute(0, 2, 3, 1)
        )

    compiled_pack = torch.compile(eager_pack, fullgraph=True)
    wrapper_pack()
    got_pack = packed.clone()
    ref_pack = torch.cat(
        (previous[:, :, 0], previous[:, :, 1], current[:, :, 0]), dim=1
    ).permute(0, 2, 3, 1).contiguous()
    add_result(
        rows,
        "vae-t1/pack_causal_cache3_nhwc",
        tuple(current.shape),
        wrapper_pack,
        native_pack,
        eager_pack,
        compiled_pack,
        got_pack,
        ref_pack,
        args,
    )

    for temporal_factor, first_chunk in ((1, False), (2, True)):
        spatial_factor = 2
        out_channels = 16 if temporal_factor == 1 else 8
        input = current if temporal_factor == 1 else torch.randn(
            (1, 64, 4, 16, 16), device="cuda", dtype=torch.bfloat16
        )
        out_t = input.shape[2] * temporal_factor - (
            temporal_factor - 1 if first_chunk else 0
        )
        out = torch.empty(
            (
                input.shape[0], out_channels, out_t,
                input.shape[3] * spatial_factor,
                input.shape[4] * spatial_factor,
            ),
            device="cuda",
            dtype=torch.bfloat16,
        )
        wrapper_up = lambda: wrapper.channel_to_space3d_bf16(
            input, out_channels, temporal_factor, spatial_factor, 1,
            first_chunk, out
        )
        native_up = lambda: native.channel_to_space3d(
            input, out_channels, temporal_factor, spatial_factor, 1,
            first_chunk, out
        )

        def eager_up():
            expanded = input[:, : out_channels * temporal_factor * 4]
            expanded = expanded.view(
                input.shape[0], out_channels, temporal_factor, 2, 2,
                input.shape[2], input.shape[3], input.shape[4],
            ).permute(0, 1, 5, 2, 6, 3, 7, 4)
            value = expanded.reshape(
                input.shape[0], out_channels,
                input.shape[2] * temporal_factor,
                input.shape[3] * 2, input.shape[4] * 2,
            )
            if first_chunk:
                value = value[:, :, temporal_factor - 1 :]
            out.copy_(value)

        compiled_up = torch.compile(eager_up, fullgraph=True)
        wrapper_up()
        got = out.clone()
        eager_up()
        ref = out.clone()
        out.copy_(got)
        add_result(
            rows,
            f"vae/channel_to_space3d-ft{temporal_factor}"
            f"-first{int(first_chunk)}",
            tuple(input.shape),
            wrapper_up,
            native_up,
            eager_up,
            compiled_up,
            got,
            ref,
            args,
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--output")
    args = parser.parse_args()
    wrapper = load_wrapper(args.backend, args.artifact)
    native = build_native()
    rows = []
    for name, shape in SHAPES.items():
        rows.extend(run_shape(wrapper, native, name, shape, args))
    rows.extend(run_cache_and_upsample(wrapper, native, args))
    for row in rows:
        print(
            f"{row.workload}: native={row.native_us:.3f}us "
            f"wrapper={row.wrapper_us:.3f}us parity={row.native_parity:.3f} "
            f"graph={row.graph_wrapper_us:.3f}/"
            f"{row.graph_native_us:.3f}us "
            f"eager={row.eager_us:.3f}us compile={row.compile_us:.3f}us "
            f"accepted={row.accepted}"
        )
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(row) for row in rows], indent=2) + "\n")
    if not all(row.accepted for row in rows):
        raise SystemExit("performance/correctness acceptance failed")


if __name__ == "__main__":
    main()
