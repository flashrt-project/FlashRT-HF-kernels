#!/usr/bin/env python3
"""Benchmark fp4-fused-ops."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
TEST_FILE = ROOT / "fp4-fused-ops" / "tests" / "test_fp4_fused_ops.py"


@dataclass
class BenchResult:
    case: str
    rows: int
    dim: int
    workload: str
    reference_us: float | None
    native_us: float | None
    flashrt_us: float
    native_graph_us: float | None
    flashrt_graph_us: float | None
    speedup: float | None
    wrapper_over_native: float | None
    graph_over_native: float | None
    status: str


def load_test_module():
    spec = importlib.util.spec_from_file_location("fp4_fused_ops_test_helpers", TEST_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helpers from {TEST_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["fp4_fused_ops_test_helpers"] = module
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


def measure_graph(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return measure(graph.replay, warmup, iters)


def bench_case(helpers, ops, native, rows: int, dim: int, warmup: int, iters: int) -> list[BenchResult]:
    make_fp16 = helpers.make_fp16
    results: list[BenchResult] = []

    residual = make_fp16((rows, dim), seed=100 + rows + dim)
    x = make_fp16((rows, dim), seed=200 + rows + dim)
    packed, sfa = ops.alloc(rows, dim)
    if dim <= 2048:
        residual_v1 = residual.clone()
        packed_v1, sfa_v1 = ops.alloc(rows, dim)
        ref_us = measure(
            lambda: ops.residual_add_rms_norm_fp4_sfa_fp16(residual_v1.copy_(residual), x, packed_v1, sfa_v1),
            warmup,
            iters,
        )
    else:
        ref_us = None
    residual_v2 = residual.clone()
    f3_us = measure(
        lambda: ops.residual_add_rms_norm_fp4_sfa_v2_fp16(residual_v2.copy_(residual), x, packed, sfa),
        warmup,
        iters,
    )
    stream = torch.cuda.current_stream().cuda_stream
    native_f3_us = measure(
        lambda: native.residual_add_rms_norm_fp4_sfa_v2_fp16(
            residual_v2.copy_(residual).data_ptr(), x.data_ptr(),
            packed.data_ptr(), sfa.data_ptr(), rows, dim, stream
        ),
        warmup,
        iters,
    )
    graph_f3_us = measure_graph(
        lambda: ops.residual_add_rms_norm_fp4_sfa_v2_fp16(residual_v2.copy_(residual), x, packed, sfa),
        warmup,
        iters,
    )
    native_graph_f3_us = measure_graph(
        lambda: native.residual_add_rms_norm_fp4_sfa_v2_fp16(
            residual_v2.copy_(residual).data_ptr(), x.data_ptr(),
            packed.data_ptr(), sfa.data_ptr(), rows, dim,
            torch.cuda.current_stream().cuda_stream
        ),
        warmup,
        iters,
    )
    results.append(
        BenchResult(
            case=f"rows{rows}_dim{dim}",
            rows=rows,
            dim=dim,
            workload="residual_add_rms_norm_fp4_sfa_v2",
            reference_us=ref_us,
            native_us=native_f3_us,
            flashrt_us=f3_us,
            native_graph_us=native_graph_f3_us,
            flashrt_graph_us=graph_f3_us,
            speedup=(ref_us / f3_us) if ref_us else None,
            wrapper_over_native=f3_us / native_f3_us,
            graph_over_native=graph_f3_us / native_graph_f3_us,
            status="v1 reference" if ref_us else "v2 only; v1 rejects this dim",
        )
    )

    merged = make_fp16((rows, dim * 2), seed=300 + rows + dim)
    packed_v1, sfa_v1 = ops.alloc(rows, dim)
    packed_v2, sfa_v2 = ops.alloc(rows, dim)
    ref_us = measure(lambda: ops.silu_mul_fp4_sfa_fp16(merged, packed_v1, sfa_v1), warmup, iters)
    f4_us = measure(lambda: ops.silu_mul_fp4_sfa_v2_fp16(merged, packed_v2, sfa_v2), warmup, iters)
    native_f4_us = measure(
        lambda: native.gate_geglu_fp4_sfa_v2_fp16(
            merged.data_ptr(), packed_v2.data_ptr(), sfa_v2.data_ptr(),
            rows, dim, stream
        ),
        warmup,
        iters,
    )
    graph_f4_us = measure_graph(
        lambda: ops.silu_mul_fp4_sfa_v2_fp16(merged, packed_v2, sfa_v2), warmup, iters
    )
    native_graph_f4_us = measure_graph(
        lambda: native.gate_geglu_fp4_sfa_v2_fp16(
            merged.data_ptr(), packed_v2.data_ptr(), sfa_v2.data_ptr(), rows, dim,
            torch.cuda.current_stream().cuda_stream
        ),
        warmup,
        iters,
    )
    results.append(
        BenchResult(
            case=f"rows{rows}_dim{dim}",
            rows=rows,
            dim=dim,
            workload="silu_mul_fp4_sfa_v2",
            reference_us=ref_us,
            native_us=native_f4_us,
            flashrt_us=f4_us,
            native_graph_us=native_graph_f4_us,
            flashrt_graph_us=graph_f4_us,
            speedup=ref_us / f4_us,
            wrapper_over_native=f4_us / native_f4_us,
            graph_over_native=graph_f4_us / native_graph_f4_us,
            status="v1 reference",
        )
    )

    inv_s = (torch.rand((dim,), device="cuda") * 0.25 + 0.875).to(torch.float16).contiguous()
    awq_us = measure(lambda: ops.silu_mul_mul_fp4_sfa_v2_fp16(merged, inv_s, packed_v2, sfa_v2), warmup, iters)
    native_awq_us = measure(
        lambda: native.gate_geglu_mul_fp4_sfa_v2_fp16(
            merged.data_ptr(), inv_s.data_ptr(), packed_v2.data_ptr(),
            sfa_v2.data_ptr(), rows, dim, stream
        ),
        warmup,
        iters,
    )
    graph_awq_us = measure_graph(
        lambda: ops.silu_mul_mul_fp4_sfa_v2_fp16(merged, inv_s, packed_v2, sfa_v2),
        warmup,
        iters,
    )
    native_graph_awq_us = measure_graph(
        lambda: native.gate_geglu_mul_fp4_sfa_v2_fp16(
            merged.data_ptr(), inv_s.data_ptr(), packed_v2.data_ptr(),
            sfa_v2.data_ptr(), rows, dim, torch.cuda.current_stream().cuda_stream
        ),
        warmup,
        iters,
    )
    results.append(
        BenchResult(
            case=f"rows{rows}_dim{dim}",
            rows=rows,
            dim=dim,
            workload="silu_mul_mul_fp4_sfa_v2",
            reference_us=None,
            native_us=native_awq_us,
            flashrt_us=awq_us,
            native_graph_us=native_graph_awq_us,
            flashrt_graph_us=graph_awq_us,
            speedup=None,
            wrapper_over_native=awq_us / native_awq_us,
            graph_over_native=graph_awq_us / native_graph_awq_us,
            status="fused AWQ producer latency",
        )
    )

    gate_packed, gate_sfa = ops.alloc(rows, dim)
    up_packed, up_sfa = ops.alloc(rows, dim)
    out_packed, out_sfa = ops.alloc(rows, dim)
    ops.silu_mul_fp4_sfa_v2_fp16(make_fp16((rows, dim * 2), seed=400 + rows + dim), gate_packed, gate_sfa)
    ops.silu_mul_fp4_sfa_v2_fp16(make_fp16((rows, dim * 2), seed=500 + rows + dim), up_packed, up_sfa)
    two_us = measure(
        lambda: ops.silu_mul_two_fp4_to_fp4(gate_packed, gate_sfa, up_packed, up_sfa, out_packed, out_sfa),
        warmup,
        iters,
    )
    native_two_us = measure(
        lambda: native.geglu_two_fp4_to_fp4(
            gate_packed.data_ptr(), gate_sfa.data_ptr(), up_packed.data_ptr(),
            up_sfa.data_ptr(), out_packed.data_ptr(), out_sfa.data_ptr(),
            rows, dim, stream
        ),
        warmup,
        iters,
    )
    graph_two_us = measure_graph(
        lambda: ops.silu_mul_two_fp4_to_fp4(gate_packed, gate_sfa, up_packed, up_sfa, out_packed, out_sfa),
        warmup,
        iters,
    )
    native_graph_two_us = measure_graph(
        lambda: native.geglu_two_fp4_to_fp4(
            gate_packed.data_ptr(), gate_sfa.data_ptr(), up_packed.data_ptr(),
            up_sfa.data_ptr(), out_packed.data_ptr(), out_sfa.data_ptr(), rows, dim,
            torch.cuda.current_stream().cuda_stream
        ),
        warmup,
        iters,
    )
    two_mul_us = measure(
        lambda: ops.silu_mul_two_mul_fp4_to_fp4(gate_packed, gate_sfa, up_packed, up_sfa, inv_s, out_packed, out_sfa),
        warmup,
        iters,
    )
    native_two_mul_us = measure(
        lambda: native.geglu_two_mul_fp4_to_fp4(
            gate_packed.data_ptr(), gate_sfa.data_ptr(), up_packed.data_ptr(),
            up_sfa.data_ptr(), inv_s.data_ptr(), out_packed.data_ptr(),
            out_sfa.data_ptr(), rows, dim, stream
        ),
        warmup,
        iters,
    )
    graph_two_mul_us = measure_graph(
        lambda: ops.silu_mul_two_mul_fp4_to_fp4(gate_packed, gate_sfa, up_packed, up_sfa, inv_s, out_packed, out_sfa),
        warmup,
        iters,
    )
    native_graph_two_mul_us = measure_graph(
        lambda: native.geglu_two_mul_fp4_to_fp4(
            gate_packed.data_ptr(), gate_sfa.data_ptr(), up_packed.data_ptr(),
            up_sfa.data_ptr(), inv_s.data_ptr(), out_packed.data_ptr(),
            out_sfa.data_ptr(), rows, dim, torch.cuda.current_stream().cuda_stream
        ),
        warmup,
        iters,
    )
    results.append(
        BenchResult(
            case=f"rows{rows}_dim{dim}",
            rows=rows,
            dim=dim,
            workload="silu_mul_two_fp4_to_fp4",
            reference_us=None,
            native_us=native_two_us,
            flashrt_us=two_us,
            native_graph_us=native_graph_two_us,
            flashrt_graph_us=graph_two_us,
            speedup=None,
            wrapper_over_native=two_us / native_two_us,
            graph_over_native=graph_two_us / native_graph_two_us,
            status="FP4-to-FP4 combiner latency",
        )
    )
    results.append(
        BenchResult(
            case=f"rows{rows}_dim{dim}",
            rows=rows,
            dim=dim,
            workload="silu_mul_two_mul_fp4_to_fp4",
            reference_us=None,
            native_us=native_two_mul_us,
            flashrt_us=two_mul_us,
            native_graph_us=native_graph_two_mul_us,
            flashrt_graph_us=graph_two_mul_us,
            speedup=None,
            wrapper_over_native=two_mul_us / native_two_mul_us,
            graph_over_native=graph_two_mul_us / native_graph_two_mul_us,
            status="FP4-to-FP4 AWQ combiner latency",
        )
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["smoke", "headline", "thor-models"], default="headline"
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    helpers = load_test_module()
    ops = helpers.load_source_ops()
    native_root = Path(os.environ.get("FLASHRT_NATIVE_ROOT", str(ROOT.parent / "official" / "FlashRT")))
    sys.path.insert(0, str(native_root))
    try:
        import flash_rt.flash_rt_fp4 as native
    finally:
        sys.path.pop(0)

    shapes = [(1, 1024), (10, 2048)] if args.mode == "smoke" else [(1, 1024), (10, 2048), (64, 2048), (128, 4096)]
    if args.mode == "thor-models":
        shapes = list(helpers.THOR_MODEL_SHAPES.values())
    results: list[BenchResult] = []
    for rows, dim in shapes:
        results.extend(bench_case(helpers, ops, native, rows, dim, args.warmup, args.iterations))

    payload = {
        "mode": args.mode,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "results": [asdict(item) for item in results],
    }
    print(json.dumps(payload, indent=2))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
