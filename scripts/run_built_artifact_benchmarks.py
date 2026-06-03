#!/usr/bin/env python3
"""Run public benchmark scripts against source extensions or built artifacts.

This local release-candidate runner preserves the public
``kernels.benchmark.Benchmark`` script format, but binds ``self.kernel`` to the
already-copied local artifact module, or to a local source-extension wrapper,
instead of downloading from the Hub.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import random
import statistics
import sys
import time
import types
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VARIANT = "torch211-cxx11-cu128-x86_64-linux"
PACKAGES = {
    "flashrt-gemm-epilogues": "flashrt_gemm_epilogues",
    "flashrt-fp8-ffn": "flashrt_fp8_ffn",
    "flashrt-vla-video": "flashrt_vla_video",
    "flashrt-nvfp4": "flashrt_nvfp4",
    "flashrt-smallm-gemm": "flashrt_smallm_gemm",
    "flashrt-fused-quant": "flashrt_fused_quant",
}


class Benchmark:
    seed: int | None = None
    device: str = "cpu"

    def __init__(self) -> None:
        self.kernel: Any = None
        self.out: Any = None

    def setup(self) -> None:
        pass


@dataclass
class WorkloadResult:
    package: str
    script: str
    benchmark: str
    workload: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    ref_ms: float | None
    speedup: float | None
    compiled_ref_ms: float | None
    speedup_vs_compiled: float | None
    compile_status: str | None
    compile_note: str | None
    verified: bool | None
    iterations: int
    backend: str = ""


def install_benchmark_shim() -> None:
    kernels_mod = types.ModuleType("kernels")
    benchmark_mod = types.ModuleType("kernels.benchmark")
    benchmark_mod.Benchmark = Benchmark
    kernels_mod.benchmark = benchmark_mod
    sys.modules["kernels"] = kernels_mod
    sys.modules["kernels.benchmark"] = benchmark_mod


def load_module(script: Path):
    spec = importlib.util.spec_from_file_location(f"_bench_{script.stem}", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    old_cwd = Path.cwd()
    os.chdir(script.parent.parent)
    try:
        spec.loader.exec_module(module)
    finally:
        os.chdir(old_cwd)
    return module


def discover_classes(module) -> list[type[Benchmark]]:
    classes: list[type[Benchmark]] = []
    for value in vars(module).values():
        if isinstance(value, type) and issubclass(value, Benchmark) and value is not Benchmark:
            classes.append(value)
    return classes


def sync(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_ms(torch, fn, iterations: int) -> list[float]:
    times: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        sync(torch)
        times.append((time.perf_counter() - start) * 1000.0)
    return times


def outputs_match(torch, got, expected) -> bool:
    if got.dtype in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bool,
    }:
        return bool((got == expected).all().item())
    if got.dtype in {torch.bfloat16, torch.float16} or expected.dtype in {
        torch.bfloat16,
        torch.float16,
    }:
        return bool(torch.allclose(got, expected, rtol=3e-2, atol=1.25e-1))
    return bool(torch.allclose(got, expected, rtol=1e-3, atol=1e-3))


def reference_compile_supported(package: str, script: Path, cls: type[Benchmark], workload: str) -> tuple[bool, str | None]:
    if package == "flashrt-nvfp4":
        return False, "layout reference uses Python loops/CPU indexing; torch.compile baseline is not meaningful"
    if package == "flashrt-smallm-gemm":
        return False, "low-bit dequant reference uses Python chunk loops; use eager reference plus strong low-bit baseline"
    return True, None


def run_workload(
    *,
    torch,
    package: str,
    script: Path,
    cls: type[Benchmark],
    method_name: str,
    kernel,
    warmup: int,
    iterations: int,
    compile_baseline: bool,
    compile_mode: str,
) -> WorkloadResult:
    workload = method_name.removeprefix("benchmark_")
    instance = cls()
    instance.kernel = kernel
    instance.device = "cuda" if torch.cuda.is_available() else "cpu"

    if instance.seed is not None:
        torch.manual_seed(instance.seed)
        random.seed(instance.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(instance.seed)

    setup = getattr(instance, f"setup_{workload}", None) or instance.setup
    setup()
    benchmark = getattr(instance, method_name)
    verify = (
        getattr(instance, f"verify_{workload}", None)
        or getattr(instance, f"reference_{workload}", None)
    )

    verified: bool | None = None
    ref_ms: float | None = None
    compiled_ref_ms: float | None = None
    compile_status: str | None = None
    compile_note: str | None = None
    if verify is not None:
        benchmark()
        sync(torch)
        reference = verify()
        sync(torch)
        verified = outputs_match(torch, instance.out, reference)
        if not verified:
            raise RuntimeError(f"{package} {cls.__name__}.{workload} verification failed")
        for _ in range(warmup):
            verify()
            sync(torch)
        ref_times = time_ms(torch, verify, max(1, min(iterations, 20)))
        ref_ms = statistics.mean(ref_times)
        if compile_baseline:
            supported, unsupported_note = reference_compile_supported(
                package, script, cls, workload
            )
            if not supported:
                compile_status = "unsupported"
                compile_note = unsupported_note
            elif not hasattr(torch, "compile"):
                compile_status = "unsupported"
                compile_note = "torch.compile is not available"
            else:
                try:
                    compiled_verify = torch.compile(verify, mode=compile_mode, fullgraph=False)
                    for _ in range(max(1, min(warmup, 5))):
                        compiled_verify()
                        sync(torch)
                    compiled_reference = compiled_verify()
                    sync(torch)
                    if not outputs_match(torch, compiled_reference, reference):
                        compile_status = "failed"
                        compile_note = "compiled reference output mismatch"
                    else:
                        compiled_times = time_ms(
                            torch,
                            compiled_verify,
                            max(1, min(iterations, 20)),
                        )
                        compiled_ref_ms = statistics.mean(compiled_times)
                        compile_status = "ok"
                        compile_note = None
                except Exception as exc:
                    compile_status = "failed"
                    compile_note = f"{type(exc).__name__}: {exc}"
    elif compile_baseline:
        compile_status = "no_reference"
        compile_note = "benchmark has no verify_* or reference_* method"

    for _ in range(warmup):
        benchmark()
        sync(torch)

    times = time_ms(torch, benchmark, iterations)
    mean = statistics.mean(times)
    std = statistics.pstdev(times) if len(times) > 1 else 0.0
    speedup = (ref_ms / mean) if ref_ms is not None and mean > 0 else None
    speedup_vs_compiled = (
        (compiled_ref_ms / mean)
        if compiled_ref_ms is not None and mean > 0
        else None
    )
    return WorkloadResult(
        package=package,
        script=str(script.relative_to(ROOT)),
        benchmark=cls.__name__,
        workload=workload,
        mean_ms=mean,
        std_ms=std,
        min_ms=min(times),
        max_ms=max(times),
        ref_ms=ref_ms,
        speedup=speedup,
        compiled_ref_ms=compiled_ref_ms,
        speedup_vs_compiled=speedup_vs_compiled,
        compile_status=compile_status,
        compile_note=compile_note,
        verified=verified,
        iterations=iterations,
    )


def error_result(
    *,
    package: str,
    backend: str,
    script: Path,
    cls: type[Benchmark],
    method_name: str,
    iterations: int,
    exc: Exception,
) -> WorkloadResult:
    return WorkloadResult(
        package=package,
        script=str(script.relative_to(ROOT)),
        benchmark=cls.__name__,
        workload=method_name.removeprefix("benchmark_"),
        mean_ms=math.nan,
        std_ms=math.nan,
        min_ms=math.nan,
        max_ms=math.nan,
        ref_ms=None,
        speedup=None,
        compiled_ref_ms=None,
        speedup_vs_compiled=None,
        compile_status="error",
        compile_note=f"{type(exc).__name__}: {exc}",
        verified=False,
        iterations=iterations,
        backend=backend,
    )


def run_package(
    package: str,
    backend: str,
    variant: str,
    warmup: int,
    iterations: int,
    compile_baseline: bool,
    compile_mode: str,
    allow_diagnostic_failures: bool,
) -> list[WorkloadResult]:
    if backend == "artifact":
        artifact = ROOT / package / "build" / variant
        if not artifact.is_dir():
            raise RuntimeError(f"missing artifact directory: {artifact}")
        sys.path.insert(0, str(artifact))
        try:
            kernel = importlib.import_module(PACKAGES[package])
        finally:
            sys.path.remove(str(artifact))
    elif backend == "source":
        from accuracy_sweep import _load_source_ops

        kernel = _load_source_ops(package)
    else:
        raise RuntimeError(f"unknown backend: {backend}")

    torch = importlib.import_module("torch")
    scripts = sorted((ROOT / package / "benchmarks").glob("benchmark*.py"))
    results: list[WorkloadResult] = []
    for script in scripts:
        module = load_module(script)
        for cls in discover_classes(module):
            methods = sorted(
                name for name in dir(cls)
                if name.startswith("benchmark_") and callable(getattr(cls, name))
            )
            for method_name in methods:
                try:
                    result = run_workload(
                        torch=torch,
                        package=package,
                        script=script,
                        cls=cls,
                        method_name=method_name,
                        kernel=kernel,
                        warmup=warmup,
                        iterations=iterations,
                        compile_baseline=compile_baseline,
                        compile_mode=compile_mode,
                    )
                    result.backend = backend
                except Exception as exc:
                    if not allow_diagnostic_failures:
                        workload = method_name.removeprefix("benchmark_")
                        raise RuntimeError(
                            f"{package} {cls.__name__}.{workload} failed verification or execution"
                        ) from exc
                    result = error_result(
                        package=package,
                        backend=backend,
                        script=script,
                        cls=cls,
                        method_name=method_name,
                        iterations=iterations,
                        exc=exc,
                    )
                results.append(result)
                speed = "" if result.speedup is None else f" speedup={result.speedup:.2f}x"
                compiled = (
                    ""
                    if result.speedup_vs_compiled is None
                    else f" compile_speedup={result.speedup_vs_compiled:.2f}x"
                )
                compile_status = (
                    ""
                    if result.compile_status is None
                    else f" compile={result.compile_status}"
                )
                verified = "" if result.verified is None else f" verified={result.verified}"
                print(
                    f"{package} {result.benchmark}.{result.workload}: "
                    f"{result.mean_ms:.4f} ms{speed}{compiled}{compile_status}{verified}",
                    flush=True,
                )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", default="all")
    parser.add_argument("--backend", choices=["artifact", "source"], default="artifact")
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--compile-baseline", action="store_true")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument(
        "--allow-diagnostic-failures",
        action="store_true",
        help="record failed diagnostic rows as nan instead of failing the run",
    )
    parser.add_argument("--output", default="internal-tests/built-artifact-benchmarks/results.json")
    args = parser.parse_args()

    install_benchmark_shim()
    packages = list(PACKAGES) if args.package == "all" else [p.strip() for p in args.package.split(",") if p.strip()]
    all_results: list[WorkloadResult] = []
    for package in packages:
        all_results.extend(
            run_package(
                package,
                args.backend,
                args.variant,
                args.warmup,
                args.iterations,
                args.compile_baseline,
                args.compile_mode,
                args.allow_diagnostic_failures,
            )
        )

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend": args.backend,
        "variant": args.variant,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "compile_baseline": args.compile_baseline,
        "compile_mode": args.compile_mode,
        "allow_diagnostic_failures": args.allow_diagnostic_failures,
        "results": [asdict(item) for item in all_results],
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
