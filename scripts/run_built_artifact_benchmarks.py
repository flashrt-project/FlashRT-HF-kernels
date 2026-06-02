#!/usr/bin/env python3
"""Run public benchmark scripts against copied built artifacts.

This local release-candidate runner preserves the public
``kernels.benchmark.Benchmark`` script format, but binds ``self.kernel`` to the
already-copied local artifact module instead of downloading from the Hub.
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VARIANT = "torch211-cxx11-cu128-x86_64-linux"
PACKAGES = {
    "flashrt-gemm-epilogues": "flashrt_gemm_epilogues",
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
    verified: bool | None
    iterations: int


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
    verify = getattr(instance, f"verify_{workload}", None)

    verified: bool | None = None
    ref_ms: float | None = None
    if verify is not None:
        benchmark()
        sync(torch)
        reference = verify()
        sync(torch)
        verified = bool(torch.allclose(instance.out, reference, atol=1e-2))
        if not verified:
            raise RuntimeError(f"{package} {cls.__name__}.{workload} verification failed")
        for _ in range(warmup):
            verify()
            sync(torch)
        ref_times = time_ms(torch, verify, max(1, min(iterations, 20)))
        ref_ms = statistics.mean(ref_times)

    for _ in range(warmup):
        benchmark()
        sync(torch)

    times = time_ms(torch, benchmark, iterations)
    mean = statistics.mean(times)
    std = statistics.pstdev(times) if len(times) > 1 else 0.0
    speedup = (ref_ms / mean) if ref_ms is not None and mean > 0 else None
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
        verified=verified,
        iterations=iterations,
    )


def run_package(package: str, warmup: int, iterations: int) -> list[WorkloadResult]:
    artifact = ROOT / package / "build" / VARIANT
    if not artifact.is_dir():
        raise RuntimeError(f"missing artifact directory: {artifact}")
    sys.path.insert(0, str(artifact))
    try:
        kernel = importlib.import_module(PACKAGES[package])
    finally:
        sys.path.remove(str(artifact))

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
                result = run_workload(
                    torch=torch,
                    package=package,
                    script=script,
                    cls=cls,
                    method_name=method_name,
                    kernel=kernel,
                    warmup=warmup,
                    iterations=iterations,
                )
                results.append(result)
                speed = "" if result.speedup is None else f" speedup={result.speedup:.2f}x"
                verified = "" if result.verified is None else f" verified={result.verified}"
                print(
                    f"{package} {result.benchmark}.{result.workload}: "
                    f"{result.mean_ms:.4f} ms{speed}{verified}",
                    flush=True,
                )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", default="all")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", default="internal-tests/built-artifact-benchmarks/results.json")
    args = parser.parse_args()

    install_benchmark_shim()
    packages = list(PACKAGES) if args.package == "all" else [p.strip() for p in args.package.split(",") if p.strip()]
    all_results: list[WorkloadResult] = []
    for package in packages:
        all_results.extend(run_package(package, args.warmup, args.iterations))

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "variant": VARIANT,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "results": [asdict(item) for item in all_results],
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
