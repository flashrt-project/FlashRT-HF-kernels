#!/usr/bin/env python3
"""Benchmark generic FP8 linear+bias projection paths."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
TEST_FILE = ROOT / "flashrt-fp8-ffn" / "tests" / "test_fp8_ffn.py"

SHAPES = {
    "decode_m1_d2048_o": (1, 2048, 2048),
    "small_m8_d1536_o": (8, 1536, 1536),
    "groot_dit_m51_qkv": (51, 1536, 4608),
    "groot_dit_m51_o": (51, 1536, 1536),
    "mid_m64_d2048_o": (64, 2048, 2048),
    "groot_backbone_m105_qkv": (105, 2048, 4096),
    "groot_backbone_m105_o": (105, 2048, 2048),
    "mid_m128_siglip_qkv": (128, 1152, 3456),
    "prefill_m256_d1536_o": (256, 1536, 1536),
    "prefill_m512_d2048_o": (512, 2048, 2048),
}


def load_test_helpers():
    spec = importlib.util.spec_from_file_location("flashrt_fp8_ffn_test_helpers", TEST_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {TEST_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_ops(backend: str, artifact: str | None):
    helpers = load_test_helpers()
    if backend == "source":
        return helpers.load_source_ops(), helpers
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return importlib.import_module("flashrt_fp8_ffn"), helpers
    finally:
        if artifact:
            sys.path.remove(artifact)


def cuda_time_us(fn, warmup: int, iters: int, rounds: int) -> float:
    samples = []
    for _ in range(rounds):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iters)
    return statistics.median(samples)


def graph_runner(fn):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph.replay


def maybe_fvk_runner(x_fp8, w_fp8, bias, out, x_scale, w_scale, M, N, K):
    try:
        from flash_rt import flash_rt_kernels as fvk
    except ImportError:
        return None, "flash_rt.flash_rt_kernels is unavailable"
    gemm = fvk.GemmRunner()
    alpha = float((x_scale * w_scale).item())
    stream = torch.cuda.current_stream().cuda_stream

    def run():
        gemm.fp8_nn_bias_bf16(
            x_fp8.data_ptr(), w_fp8.data_ptr(), out.data_ptr(), bias.data_ptr(),
            M, N, K, alpha, stream,
        )

    try:
        run()
    except RuntimeError as error:
        return None, str(error)
    return run, None


def maybe_fvk_fp16_runners(
    x_fp8, w_fp8, bias_fp16, out_fp16, x_scale, w_scale, M, N, K
):
    try:
        from flash_rt import flash_rt_kernels as fvk
    except ImportError:
        return None, None, "flash_rt.flash_rt_kernels is unavailable"
    gemm = fvk.GemmRunner()
    alpha = float((x_scale * w_scale).item())
    stream = torch.cuda.current_stream().cuda_stream

    def fused():
        gemm.fp8_nn_bias(
            x_fp8.data_ptr(), w_fp8.data_ptr(), out_fp16.data_ptr(),
            bias_fp16.data_ptr(), M, N, K, alpha, stream,
        )

    def decomposed():
        gemm.fp8_descale_fp16(
            x_fp8.data_ptr(), w_fp8.data_ptr(), out_fp16.data_ptr(), M, N, K,
            x_scale.data_ptr(), w_scale.data_ptr(), stream,
        )
        fvk.add_bias_fp16(
            out_fp16.data_ptr(), bias_fp16.data_ptr(), M, N, stream
        )

    try:
        fused()
        fused_runner = fused
        fused_error = None
    except RuntimeError as error:
        fused_runner = None
        fused_error = str(error)
    decomposed()
    return fused_runner, decomposed, fused_error


def metrics(got: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    got_f = got.float().flatten()
    exp_f = expected.float().flatten()
    diff = (got_f - exp_f).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "p99_abs": float(torch.quantile(diff, 0.99).item()),
        "cosine": float(torch.nn.functional.cosine_similarity(got_f, exp_f, dim=0).item()),
    }


def run(args) -> dict:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(20260718)
    ops, helpers = load_ops(args.backend, args.artifact)
    labels = list(SHAPES) if args.shapes == "all" else args.shapes.split(",")
    results = []

    for label in labels:
        if label not in SHAPES:
            raise ValueError(f"unknown shape {label!r}")
        M, K, N = SHAPES[label]
        x_bf16 = torch.randn((M, K), device="cuda", dtype=torch.bfloat16) * 0.25
        w_bf16 = torch.randn((N, K), device="cuda", dtype=torch.bfloat16) * (K**-0.5)
        bias = torch.randn((N,), device="cuda", dtype=torch.bfloat16) * 0.01
        x_scale = (
            x_bf16.float().abs().max() / (0.9 * helpers.fp8_max())
        ).clamp_min(1e-6).reshape(1)
        w_scale = (
            w_bf16.float().abs().max() / (0.9 * helpers.fp8_max())
        ).clamp_min(1e-6).reshape(1)
        x_fp8 = helpers.quantize_fp8_reciprocal(x_bf16, x_scale)
        w_fp8 = helpers.quantize_fp8(w_bf16, w_scale)
        out = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
        region_out = torch.empty_like(out)
        input_fp8 = torch.empty_like(x_fp8)
        fvk_out = torch.empty_like(out)
        fvk_fp16_out = torch.empty((M, N), device="cuda", dtype=torch.float16)
        bias_fp16 = bias.to(torch.float16)

        def package_fp8():
            ops.fp8_linear_bias_bf16(
                x_fp8, w_fp8, bias, x_scale, w_scale, out=out
            )

        def package_region():
            ops.bf16_fp8_linear_bias_bf16(
                x_bf16, w_fp8, bias, x_scale, w_scale,
                input_fp8=input_fp8, out=region_out, pad_to=M,
            )

        reference = helpers.ref_linear_bias(
            x_fp8, w_fp8, bias, x_scale, w_scale
        )
        package_fp8()
        package_metrics = metrics(out, reference)
        if package_metrics["p99_abs"] > 0.015625 or package_metrics["cosine"] < 0.9999:
            raise AssertionError(f"{label} package correctness failed: {package_metrics}")
        package_region()
        if not torch.equal(region_out, out) or not torch.equal(input_fp8, x_fp8):
            raise AssertionError(f"{label} BF16 region does not match FP8 entry")

        graph = graph_runner(package_region)
        if args.compare_fvk:
            fvk, fvk_error = maybe_fvk_runner(
                x_fp8, w_fp8, bias, fvk_out, x_scale, w_scale, M, N, K
            )
        else:
            fvk, fvk_error = None, "not requested"
        if args.compare_fvk:
            fvk_fp16_fused, fvk_fp16_decomposed, fvk_fp16_error = (
                maybe_fvk_fp16_runners(
                    x_fp8, w_fp8, bias_fp16, fvk_fp16_out, x_scale, w_scale,
                    M, N, K,
                )
            )
        else:
            fvk_fp16_fused = None
            fvk_fp16_decomposed = None
            fvk_fp16_error = "not requested"
        if fvk is not None:
            fvk()
            fvk_metrics = metrics(fvk_out, reference)
            if fvk_metrics["p99_abs"] > 0.015625 or fvk_metrics["cosine"] < 0.9999:
                raise AssertionError(f"{label} FVK correctness failed: {fvk_metrics}")
        else:
            fvk_metrics = None

        def eager(value):
            return torch.addmm(bias, value, w_bf16.T)

        compiled = torch.compile(eager, fullgraph=True) if args.compile_baseline else None
        if compiled is not None:
            compiled_out = compiled(x_bf16)
            changed_out = compiled(x_bf16 + torch.ones_like(x_bf16) * 0.125)
            torch.cuda.synchronize()
            if torch.equal(compiled_out, changed_out):
                raise AssertionError(
                    f"{label} compiled baseline did not respond to changed input"
                )

        package_us = cuda_time_us(package_fp8, args.warmup, args.iters, args.rounds)
        region_us = cuda_time_us(package_region, args.warmup, args.iters, args.rounds)
        graph_us = cuda_time_us(graph, args.warmup, args.iters, args.rounds)
        eager_call = lambda: eager(x_bf16)
        compiled_call = (lambda: compiled(x_bf16)) if compiled is not None else None
        eager_us = cuda_time_us(eager_call, args.warmup, args.iters, args.rounds)
        compiled_us = (
            cuda_time_us(compiled_call, args.warmup, args.iters, args.rounds)
            if compiled_call is not None else None
        )
        fvk_us = (
            cuda_time_us(fvk, args.warmup, args.iters, args.rounds)
            if fvk is not None else None
        )
        fvk_fp16_fused_us = (
            cuda_time_us(
                fvk_fp16_fused, args.warmup, args.iters, args.rounds
            )
            if fvk_fp16_fused is not None else None
        )
        fvk_fp16_decomposed_us = (
            cuda_time_us(
                fvk_fp16_decomposed, args.warmup, args.iters, args.rounds
            )
            if fvk_fp16_decomposed is not None else None
        )
        row = {
            "shape": label,
            "M": M,
            "K": K,
            "N": N,
            "fp8_linear_bias_us": package_us,
            "bf16_region_us": region_us,
            "bf16_region_graph_us": graph_us,
            "torch_bf16_eager_us": eager_us,
            "torch_bf16_compiled_us": compiled_us,
            "fvk_fp8_nn_bias_bf16_us": fvk_us,
            "fvk_status": "ok" if fvk is not None else "unsupported",
            "fvk_error": fvk_error,
            "fvk_fp16_fused_us": fvk_fp16_fused_us,
            "fvk_fp16_decomposed_us": fvk_fp16_decomposed_us,
            "fvk_fp16_fused_error": fvk_fp16_error,
            "package_vs_eager": eager_us / package_us,
            "region_vs_eager": eager_us / region_us,
            "graph_vs_eager": eager_us / graph_us,
            "package_vs_fvk": fvk_us / package_us if fvk_us is not None else None,
            "package_vs_fvk_fp16_fused": (
                fvk_fp16_fused_us / package_us
                if fvk_fp16_fused_us is not None else None
            ),
            "package_vs_fvk_fp16_decomposed": (
                fvk_fp16_decomposed_us / package_us
                if fvk_fp16_decomposed_us is not None else None
            ),
            "package_correctness": package_metrics,
            "fvk_correctness": fvk_metrics,
        }
        results.append(row)
        print(json.dumps(row, sort_keys=True))

    return {
        "backend": args.backend,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "warmup": args.warmup,
        "iters": args.iters,
        "rounds": args.rounds,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--shapes", default="all")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--compile-baseline", action="store_true")
    parser.add_argument("--compare-fvk", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    payload = run(args)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
