#!/usr/bin/env python3
"""Benchmark fp4-gemm."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
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
TEST_FILE = ROOT / "fp4-gemm" / "tests" / "test_fp4_gemm.py"


@dataclass
class BenchResult:
    shape: str
    M: int
    N: int
    K: int
    variant: int
    native_us: float
    flashrt_us: float
    torch_eager_us: float
    torch_compile_us: float
    speedup_vs_eager: float
    speedup_vs_compile: float
    wrapper_over_native: float
    max_abs: float
    mean_abs: float
    p99_abs: float
    cosine: float
    status: str


def load_helpers():
    spec = importlib.util.spec_from_file_location("fp4_gemm_test_helpers", TEST_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helpers from {TEST_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["fp4_gemm_test_helpers"] = module
    spec.loader.exec_module(module)
    return module


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


def bench_case(helpers, ops, native, name: str, shape: tuple[int, int, int], warmup: int, iters: int) -> list[BenchResult]:
    m, n, k = shape
    a_packed, b_packed, sfa, sfb, expected = helpers.prepare_quantized(ops, m, n, k)
    a_deq = torch.empty((m, k), device="cuda", dtype=torch.float16)
    b_deq = torch.empty((n, k), device="cuda", dtype=torch.float16)
    ops.dequantize_fp4_sfa_fp16(a_packed, sfa, a_deq, False)
    ops.dequantize_fp4_sfa_fp16(b_packed, sfb, b_deq, True)
    torch.cuda.synchronize()

    def torch_ref():
        return (a_deq.float() @ b_deq.float().T).to(torch.bfloat16)

    torch_eager_us = measure(torch_ref, warmup, iters)
    compiled_ref = torch.compile(torch_ref, mode="max-autotune-no-cudagraphs")
    torch_compile_us = measure(compiled_ref, warmup, iters)
    stream = torch.cuda.current_stream().cuda_stream
    results: list[BenchResult] = []
    variants = (-1, 0, 1, 2) if torch.cuda.get_device_capability(0) == (11, 0) else (0, 1, 2)
    for variant in variants:
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        ops.nvfp4_gemm_bf16(a_packed, b_packed, sfa, sfb, out, 1.0, variant)
        torch.cuda.synchronize()
        max_abs, mean_abs, p99_abs, cosine = helpers.metrics(out, expected)
        flashrt_us = measure(
            lambda: ops.nvfp4_gemm_bf16(a_packed, b_packed, sfa, sfb, out, 1.0, variant),
            warmup,
            iters,
        )
        native_variant = variant
        native_us = float("nan")
        if native is not None:
            if native_variant < 0:
                native_variant = helpers.select_sm110_variant(shape)
            native_function = (
                native.fp4_w4a16_gemm_sm120_bf16out
                if native_variant == 0
                else native.fp4_w4a16_gemm_sm120_bf16out_widen
                if native_variant == 1
                else native.fp4_w4a16_gemm_sm120_bf16out_pingpong
            )
            native_us = measure(
                lambda: native_function(
                    a_packed.data_ptr(),
                    b_packed.data_ptr(),
                    out.data_ptr(),
                    m,
                    n,
                    k,
                    sfa.data_ptr(),
                    sfb.data_ptr(),
                    1.0,
                    stream,
                ),
                warmup,
                iters,
            )
        results.append(
            BenchResult(
                shape=name,
                M=m,
                N=n,
                K=k,
                variant=variant,
                native_us=native_us,
                flashrt_us=flashrt_us,
                torch_eager_us=torch_eager_us,
                torch_compile_us=torch_compile_us,
                speedup_vs_eager=torch_eager_us / flashrt_us,
                speedup_vs_compile=torch_compile_us / flashrt_us,
                wrapper_over_native=flashrt_us / native_us,
                max_abs=max_abs,
                mean_abs=mean_abs,
                p99_abs=p99_abs,
                cosine=cosine,
                status="ok",
            )
        )
    return results


def bench_bf16_producer(ops, native, k: int, warmup: int, iters: int):
    x = torch.randn((1, k), device="cuda", dtype=torch.bfloat16)
    direct_packed, direct_sfa = ops.alloc_fp4(1, k)
    compat_packed, compat_sfa = ops.alloc_fp4(1, k)
    native_packed, native_sfa = ops.alloc_fp4(1, k)
    stream = torch.cuda.current_stream().cuda_stream

    def direct():
        ops.quantize_fp4_sfa_bf16(
            x, direct_packed, direct_sfa, False
        )

    def compat():
        ops.quantize_fp4_sfa_fp16(
            x.to(torch.float16), compat_packed, compat_sfa, False
        )

    def native_direct():
        native.quantize_bf16_to_nvfp4_swizzled(
            x.data_ptr(), native_packed.data_ptr(), native_sfa.data_ptr(),
            1, k, stream,
        )

    direct()
    compat()
    torch.cuda.synchronize()
    direct_us = measure(direct, warmup, iters)
    compat_us = measure(compat, warmup, iters)
    native_us = float("nan") if native is None else measure(native_direct, warmup, iters)
    return {
        "M": 1,
        "K": k,
        "direct_bf16_us": direct_us,
        "cast_plus_fp16_us": compat_us,
        "native_bf16_us": native_us,
        "speedup_vs_cast_plus_fp16": compat_us / direct_us,
        "wrapper_over_native": direct_us / native_us,
        "packed_exact_vs_fp16_contract": bool(
            torch.equal(direct_packed, compat_packed)
        ),
        "note": "native_bf16 uses a distinct FlashRT quantization strategy",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument(
        "--mode", choices=["smoke", "headline", "thor-models"], default="headline"
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--max-mem-gb", type=float, default=30.0)
    args = parser.parse_args()
    _apply_mem_cap(args.max_mem_gb)

    helpers = load_helpers()
    native_root = Path(
        os.environ.get("FLASHRT_NATIVE_ROOT", str(ROOT.parent / "official" / "FlashRT"))
    )
    native = None
    if native_root.is_dir():
        sys.path.insert(0, str(native_root))
        try:
            import flash_rt.flash_rt_kernels as native
        except Exception:
            native = None
        finally:
            sys.path.pop(0)
    ops = (
        helpers.load_source_ops()
        if args.backend == "source"
        else helpers.load_installed_ops(args.artifact)
    )
    shapes = {
        "small_m16_n128_k128": (16, 128, 128),
        "small_m32_n256_k256": (32, 256, 256),
        "mlp_tile_m64_n512_k512": (64, 512, 512),
        "groot_dit_projection": (51, 1536, 1536),
        "vla_projection": (105, 2048, 2048),
        "motus_up": (360, 14336, 3072),
        "motus_down": (360, 3072, 14336),
    }
    if args.mode == "smoke":
        shapes = {"small_m16_n128_k128": shapes["small_m16_n128_k128"]}
    elif args.mode == "thor-models":
        shapes = dict(helpers.SM110_SHAPES)
    results: list[BenchResult] = []
    for name, shape in shapes.items():
        results.extend(
            bench_case(
                helpers, ops, native, name, shape, args.warmup, args.iterations
            )
        )
    producer_results = [
        bench_bf16_producer(ops, native, k, args.warmup, args.iterations)
        for k in (5120, 6144, 17408)
    ]
    payload = {
        "mode": args.mode,
        "backend": args.backend,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "results": [asdict(item) for item in results],
        "bf16_producer_results": producer_results,
    }
    print(json.dumps(payload, indent=2))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
