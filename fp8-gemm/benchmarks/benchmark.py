#!/usr/bin/env python3
"""Benchmark fp8-gemm."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


def _apply_mem_cap(max_mem_gb: float = 30.0) -> None:
    if not torch.cuda.is_available() or max_mem_gb <= 0:
        return
    total = torch.cuda.get_device_properties(0).total_memory
    cap = int(max_mem_gb * 1024**3)
    if total <= 0 or cap >= total:
        return
    torch.cuda.set_per_process_memory_fraction(cap / total)


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "fp8-gemm"
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
    "decode_m1_k4096_n2048": (1, 4096, 2048),
    "decode_m1_k4096_n8192": (1, 4096, 8192),
    "small_m16_k4096_n4096": (16, 4096, 4096),
    "small_m32_k4096_n8192": (32, 4096, 8192),
    "small_m64_k512_n1024": (64, 512, 1024),
    "pi05_action_qkv": (51, 2048, 2560),
    "pi05_action_o": (51, 2048, 2048),
    "pi05_action_gate_up": (51, 2048, 16384),
    "pi05_action_down": (51, 8192, 2048),
    "groot_dit_qkv": (51, 1536, 4608),
    "groot_n17_llm_o": (277, 2048, 2048),
    "groot_n17_llm_gate_up": (277, 2048, 16384),
    "groot_n17_llm_down": (277, 8192, 2048),
    "groot_n17_vit_o": (1024, 1024, 1024),
    "cosmos_edge_action": (64, 2048, 9216),
    "lingbot_vision_o": (1024, 1280, 1280),
    "lingbot_action_gate_up": (105, 2048, 16384),
}

MODES = {
    "smoke": ["decode_m1_k4096_n2048", "small_m16_k4096_n4096"],
    "headline": [
        "decode_m1_k4096_n2048",
        "pi05_action_qkv",
        "pi05_action_gate_up",
        "pi05_action_down",
        "groot_n17_llm_o",
        "cosmos_edge_action",
        "lingbot_action_gate_up",
    ],
    "thor-full": list(SHAPES),
}


@dataclass
class Result:
    shape: str
    M: int
    K: int
    N: int
    variant: int
    tile: str
    flashrt_us: float
    flashrt_graph_us: float | None
    package_best_graph_tile: str | None
    package_best_graph_us: float | None
    auto_over_package_best: float | None
    package_tiles_us: dict[str, dict[str, float]] | None
    native_best_tile: str | None
    native_best_us: float | None
    native_best_graph_tile: str | None
    native_best_graph_us: float | None
    native_tiles_us: dict[str, dict[str, float]] | None
    wrapper_vs_native: float | None
    graph_vs_native: float | None
    torch_eager_us: float
    torch_compile_us: float | None
    speedup_vs_eager: float
    speedup_vs_compile: float | None
    max_abs: float
    mean_abs: float
    p99_abs: float
    cosine: float
    status: str


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self._ops = getattr(torch.ops, namespace)

    @staticmethod
    def select_fp8_linear_tile(m: int, n: int, k: int, variant: int = 0) -> str:
        return select_tile(m, n, k, variant)

    def fp8_linear_bf16(self, x, w, alpha=1.0, out=None, variant=0):
        if out is None:
            out = torch.empty((x.shape[0], w.shape[0]), device=x.device, dtype=torch.bfloat16)
        self._ops.fp8_linear_bf16(x, w, float(alpha), int(variant), out)
        return out


def _current_arch_list() -> str:
    major, minor = torch.cuda.get_device_capability(0)
    if (major, minor) == (11, 0):
        return "11.0a"
    if major >= 12:
        return "12.0a"
    return f"{major}.{minor}"


def load_source_ops() -> SourceOps:
    from torch.utils.cpp_extension import load

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _current_arch_list())
    namespace = "fp8_gemm_source_bench"
    capability = torch.cuda.get_device_capability(0)
    cutlass_include = Path(os.environ.get("CUTLASS_INCLUDE", ""))
    if capability == (11, 0):
        if not (cutlass_include / "cutlass" / "cutlass.h").is_file():
            raise RuntimeError("set CUTLASS_INCLUDE for the SM110 source benchmark")
        cuda_sources = [str(PACKAGE / "csrc" / "cutlass_sm110_fp8_gemm.cu")]
        source_define = "-DFLASHRT_FP8_GEMM_SOURCE_SM110_ONLY"
        extra_includes = [
            str(cutlass_include),
            str(cutlass_include.parent / "tools" / "util" / "include"),
        ]
    else:
        cuda_sources = [
            str(PACKAGE / "csrc" / "fp8_gemv_m1_sm120.cu"),
            str(PACKAGE / "csrc" / "fp8_smallM_handtuned_sm120.cu"),
            str(PACKAGE / "csrc" / "fp8_smallM_handtuned_ldmatrix_sm120.cu"),
        ]
        source_define = "-DFLASHRT_FP8_GEMM_SOURCE_SM120_ONLY"
        extra_includes = []
    load(
        name=namespace,
        sources=[str(PACKAGE / "torch-ext" / "torch_binding.cpp"), *cuda_sources],
        extra_include_paths=[
            str(PACKAGE / "csrc"), str(REGISTRATION_INCLUDE), *extra_includes
        ],
        extra_cflags=["-O3", "-DNDEBUG", "-DCUDA_KERNEL", source_define],
        extra_cuda_cflags=[
            "-O3", "-DNDEBUG", "--expt-relaxed-constexpr", "--use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-DCUDA_KERNEL", source_define
        ],
        verbose=False,
    )
    return SourceOps(namespace)


def load_installed_ops(artifact: str | None):
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return importlib.import_module("fp8_gemm")
    finally:
        if artifact:
            sys.path.remove(artifact)


def select_tile(m: int, n: int, k: int, variant: int = 0) -> str:
    if torch.cuda.get_device_capability(0) == (11, 0):
        forced = {1: "sm110_sq_bf16", 2: "sm110_t1_bf16", 3: "sm110_wide_bf16"}
        if variant not in {0, *forced}:
            raise RuntimeError("SM110 variant must be in [0, 3]")
        if variant:
            return forced[variant]
        if n >= 8 * k:
            return "sm110_wide_bf16"
        if m >= 128 and k >= 4 * n:
            return "sm110_sq_bf16"
        if n == k and m >= 512:
            return "sm110_sq_bf16" if k <= 1024 else "sm110_wide_bf16"
        if n == k and m >= 128:
            return "sm110_wide_bf16"
        return "sm110_t1_bf16"
    if m == 1:
        if variant == 4:
            return "gemv_fp8_m1_w4"
        if variant == 8:
            return "gemv_fp8_m1_w8"
        if variant == 16:
            return "gemv_fp8_m1_w16"
        if n <= 2048:
            return "gemv_fp8_m1_w4"
        if n <= 8192:
            return "gemv_fp8_m1_w8"
        return "gemv_fp8_m1_w16"
    if m <= 16:
        if k % 256 == 0:
            return "ld_fp8_gemm_16x128x256_w4" if n % 128 == 0 else "ld_fp8_gemm_16x64x256_w4"
        if n % 256 == 0:
            return "ld_fp8_gemm_16x256x128_w8"
        if n % 192 == 0:
            return "ld_fp8_gemm_16x192x128_w4"
        if n % 128 == 0:
            return "ld_fp8_gemm_16x128x128_w4"
        return "ld_fp8_gemm_16x64x128_w4"
    if m <= 32:
        if k % 256 == 0:
            return "ld_fp8_gemm_32x128x256_w4" if n % 128 == 0 else "ld_fp8_gemm_32x64x256_w4"
        if n % 192 == 0:
            return "ld_fp8_gemm_32x192x128_w4"
        if n % 128 == 0:
            return "ld_fp8_gemm_32x128x128_w4"
        return "ld_fp8_gemm_32x64x128_w4"
    if m <= 64:
        if k % 256 == 0:
            return "ld_fp8_gemm_64x128x256_w4" if n % 128 == 0 else "ld_fp8_gemm_64x64x256_w4"
        if n % 128 == 0:
            return "ld_fp8_gemm_64x128x128_w4"
        return "ld_fp8_gemm_64x64x128_w4"
    if m <= 64:
        if k % 256 == 0:
            return "ld_fp8_gemm_64x128x256_w4" if n % 128 == 0 else "ld_fp8_gemm_64x64x256_w4"
        if n % 128 == 0:
            return "ld_fp8_gemm_64x128x128_w4"
        return "ld_fp8_gemm_64x64x128_w4"
    raise RuntimeError("unsupported M")


def make_inputs(m: int, k: int, n: int, seed: int):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    x = (torch.randn((m, k), device="cuda", generator=gen) * 0.25).to(torch.bfloat16).to(torch.float8_e4m3fn)
    w = (torch.randn((n, k), device="cuda", generator=gen) * 0.25).to(torch.bfloat16).to(torch.float8_e4m3fn)
    return x, w


def ref_fn(x, w):
    return (x.float() @ w.float().T).to(torch.bfloat16)


def measure(fn, warmup: int, iters: int) -> float:
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
    return float(start.elapsed_time(end) * 1000.0 / iters)


def measure_median(fn, warmup: int, iters: int, rounds: int = 5) -> float:
    """Reduce clock/order bias without hiding Python-side launch behavior."""
    return float(statistics.median(measure(fn, warmup, iters) for _ in range(rounds)))


def capture_graph(fn, warmup: int) -> torch.cuda.CUDAGraph:
    """Capture one static invocation and retain the graph for paired timing."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    return graph


def measure_group(functions, warmup: int, iters: int, rounds: int = 7):
    """Measure one launch per candidate per round to balance Thor DVFS drift."""
    names = list(functions)
    samples = {name: [] for name in names}
    for _ in range(warmup):
        for fn in functions.values():
            fn()
    torch.cuda.synchronize()
    sample_count = max(iters, rounds * 16)
    event_pairs = {name: [] for name in names}
    for round_index in range(sample_count):
        offset = round_index % len(names)
        for name in names[offset:] + names[:offset]:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            functions[name]()
            end.record()
            event_pairs[name].append((start, end))
    torch.cuda.synchronize()
    for name, pairs in event_pairs.items():
        samples[name] = [start.elapsed_time(end) * 1000.0 for start, end in pairs]
    medians = {
        name: float(statistics.median(values)) for name, values in samples.items()
    }
    return medians, samples


def metrics(got, expected):
    diff = (got.float() - expected.float()).abs().flatten()
    return (
        float(diff.max().item()),
        float(diff.mean().item()),
        float(torch.quantile(diff, 0.99).item()),
        float(torch.nn.functional.cosine_similarity(got.float().flatten(), expected.float().flatten(), dim=0).item()),
    )


def load_native_reference():
    root = os.environ.get("FLASHRT_NATIVE_ROOT")
    if not root:
        return None
    sys.path.insert(0, root)
    try:
        return importlib.import_module("flash_rt.flash_rt_kernels")
    finally:
        sys.path.remove(root)


def bench_case(ops, native, name: str, shape: tuple[int, int, int], variant: int, warmup: int, iters: int, compile_ref: bool):
    m, k, n = shape
    x, w = make_inputs(m, k, n, seed=3000 + m + k + n + variant)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    expected = ref_fn(x, w)
    got = ops.fp8_linear_bf16(x, w, out=out, variant=variant)
    torch.cuda.synchronize()
    max_abs, mean_abs, p99_abs, cos = metrics(got, expected)
    tile = ops.select_fp8_linear_tile(m, n, k, variant)

    wrapper_invoke = lambda: ops.fp8_linear_bf16(x, w, out=out, variant=variant)
    eager_functions = {"wrapper": wrapper_invoke}
    graph_objects = {"wrapper": capture_graph(wrapper_invoke, warmup)}
    package_forced_tiles = {}
    if torch.cuda.get_device_capability(0) == (11, 0):
        for forced_variant, forced_tile in {
            1: "sm110_sq_bf16",
            2: "sm110_t1_bf16",
            3: "sm110_wide_bf16",
        }.items():
            invoke = lambda forced_variant=forced_variant: ops.fp8_linear_bf16(
                x, w, out=out, variant=forced_variant
            )
            invoke()
            torch.cuda.synchronize()
            fmax, fmean, fp99, fcos = metrics(out, expected)
            if fmax > 0.5 or fmean > 0.02 or fp99 > 0.25 or fcos < 0.999:
                raise RuntimeError(
                    f"package {forced_tile} failed correctness for {name}: "
                    f"{fmax=}, {fmean=}, {fp99=}, {fcos=}"
                )
            key = f"package:{forced_tile}"
            package_forced_tiles[forced_tile] = invoke
            eager_functions[key] = invoke
            graph_objects[key] = capture_graph(invoke, warmup)
    package_best_graph_tile = None
    package_best_graph_us = None
    package_tiles_us = None
    native_best_tile = None
    native_best_us = None
    native_best_graph_tile = None
    native_best_graph_us = None
    native_tiles_us = None
    native_invokes = {}
    wrapper_vs_native = None
    graph_vs_native = None
    auto_over_package_best = None
    if native is not None and torch.cuda.get_device_capability(0) == (11, 0):
        native_out = torch.empty_like(out)
        candidates = [
            ("sm110_sq_bf16", native.cutlass_fp8_sq_bf16out),
            ("sm110_t1_bf16", native.cutlass_fp8_t1_bf16out),
            ("sm110_wide_bf16", native.cutlass_fp8_wide_bf16out),
        ]
        for tile_name, fn in candidates:
            invoke = lambda fn=fn: fn(
                x.data_ptr(), w.data_ptr(), native_out.data_ptr(),
                m, n, k, 1.0, 0.0,
                int(torch.cuda.current_stream().cuda_stream),
            )
            rc = invoke()
            if rc != 0:
                continue
            torch.cuda.synchronize()
            nmax, nmean, np99, ncos = metrics(native_out, expected)
            if nmax > 0.5 or nmean > 0.02 or np99 > 0.25 or ncos < 0.999:
                raise RuntimeError(
                    f"native {tile_name} failed correctness for {name}: "
                    f"{nmax=}, {nmean=}, {np99=}, {ncos=}"
                )
            key = f"native:{tile_name}"
            native_invokes[tile_name] = invoke
            eager_functions[key] = invoke
            graph_objects[key] = capture_graph(invoke, warmup)
    if package_forced_tiles:
        eager_times, eager_samples = measure_group(eager_functions, warmup, iters)
        graph_times, graph_samples = measure_group(
            {
                key: graph.replay
                for key, graph in graph_objects.items()
                if key != "wrapper"
            },
            warmup,
            iters,
        )
        flashrt_us = eager_times["wrapper"]
        package_tiles_us = {
            tile_name: {
                "eager": eager_times[f"package:{tile_name}"],
                "graph": graph_times[f"package:{tile_name}"],
            }
            for tile_name in package_forced_tiles
        }
        package_best_graph_us, package_best_graph_tile = min(
            (times["graph"], tile_name)
            for tile_name, times in package_tiles_us.items()
        )
        # Auto and its matching forced variant resolve to the same native
        # function. Use that single graph measurement for the tile gate; timing
        # duplicate graph objects is vulnerable to Thor DVFS order bias.
        flashrt_graph_us = package_tiles_us[tile]["graph"]
        auto_over_package_best = float(statistics.median(
            selected / best
            for selected, best in zip(
                graph_samples[f"package:{tile}"],
                graph_samples[f"package:{package_best_graph_tile}"],
            )
        ))
        if native_invokes:
            native_tiles_us = {
                tile_name: {
                    "eager": eager_times[f"native:{tile_name}"],
                    "graph": graph_times[f"native:{tile_name}"],
                }
                for tile_name in native_invokes
            }
            native_best_us, native_best_tile = min(
                (times["eager"], tile_name)
                for tile_name, times in native_tiles_us.items()
            )
            native_best_graph_us, native_best_graph_tile = min(
                (times["graph"], tile_name)
                for tile_name, times in native_tiles_us.items()
            )
            wrapper_vs_native = float(statistics.median(
                wrapper / native_sample
                for wrapper, native_sample in zip(
                    eager_samples["wrapper"],
                    eager_samples[f"native:{native_best_tile}"],
                )
            ))
            graph_vs_native = float(statistics.median(
                package_sample / native_sample
                for package_sample, native_sample in zip(
                    graph_samples[f"package:{tile}"],
                    graph_samples[f"native:{native_best_graph_tile}"],
                )
            ))
    else:
        flashrt_us = measure_median(wrapper_invoke, warmup, iters)
        flashrt_graph_us = measure_median(
            graph_objects["wrapper"].replay, warmup, iters
        )
    tile_pass = auto_over_package_best is None or auto_over_package_best <= 1.10
    eager_us = measure(lambda: ref_fn(x, w), warmup, iters)
    compile_us = None
    if compile_ref:
        try:
            compiled = torch.compile(ref_fn, fullgraph=True)
            compiled(x, w)
            torch.cuda.synchronize()
            compile_us = measure(lambda: compiled(x, w), warmup, iters)
        except Exception:
            compile_us = None

    return Result(
        shape=name,
        M=m,
        K=k,
        N=n,
        variant=variant,
        tile=tile,
        flashrt_us=flashrt_us,
        flashrt_graph_us=flashrt_graph_us,
        package_best_graph_tile=package_best_graph_tile,
        package_best_graph_us=package_best_graph_us,
        auto_over_package_best=auto_over_package_best,
        package_tiles_us=package_tiles_us,
        native_best_tile=native_best_tile,
        native_best_us=native_best_us,
        native_best_graph_tile=native_best_graph_tile,
        native_best_graph_us=native_best_graph_us,
        native_tiles_us=native_tiles_us,
        wrapper_vs_native=wrapper_vs_native,
        graph_vs_native=graph_vs_native,
        torch_eager_us=eager_us,
        torch_compile_us=compile_us,
        speedup_vs_eager=eager_us / flashrt_us,
        speedup_vs_compile=(compile_us / flashrt_us) if compile_us else None,
        max_abs=max_abs,
        mean_abs=mean_abs,
        p99_abs=p99_abs,
        cosine=cos,
        status=(
            "pass"
            if max_abs <= 0.5 and p99_abs <= 0.25 and cos >= 0.999 and tile_pass
            else "fail"
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--mode", choices=sorted(MODES), default="smoke")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--compile-ref", action="store_true")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--max-mem-gb", type=float, default=30.0)
    args = parser.parse_args()
    _apply_mem_cap(args.max_mem_gb)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    capability = torch.cuda.get_device_capability(0)
    if capability not in {(11, 0), (12, 0)}:
        raise SystemExit("fp8-gemm benchmark requires SM110 or SM120")

    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    native = load_native_reference()
    rows: list[Result] = []
    for name in MODES[args.mode]:
        shape = SHAPES[name]
        variants = [0]
        if shape[0] == 1 and capability == (12, 0):
            variants = [0, 4, 8, 16]
        for variant in variants:
            rows.append(bench_case(
                ops, native, name, shape, variant,
                args.warmup, args.iterations, args.compile_ref,
            ))

    payload = {"rows": [asdict(row) for row in rows]}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if any(row.status != "pass" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
