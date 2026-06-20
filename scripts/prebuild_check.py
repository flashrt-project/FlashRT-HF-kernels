#!/usr/bin/env python3
"""Pre-build checks for FlashRT HF kernel packages.

This script intentionally does not run `kernel-builder build`. It verifies the
repository state before a release build window so small edits can be checked
quickly without paying the full Nix/build cost.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V1_PACKAGES = [
    "flashrt-gemm-epilogues",
    "flashrt-fp8-ffn",
    "flashrt-vla-video",
    "flashrt-nvfp4",
    "flashrt-smallm-gemm",
    "flashrt-fused-quant",
]
V2_CANDIDATE_PACKAGES = [
    "flashrt-fp8-swiglu-ffn",
    "flashrt-residual-norm-quant",
    "flashrt-qkv-cache-rope",
    "flashrt-vla-residual-gates",
    "flashrt-adaptive-norms",
    "flashrt-spatiotemporal-layout",
    "MiniMaxAI-msa-blackwell",
    "vl-transformer-primitives",
    "diffusion-step-ops",
    "turboquant-kv",
    "linear-attention-primitives",
    "world-model-conv",
]
REQUIRED_DOCS = [
    "docs/benchmark-baselines.md",
    "docs/correctness-gating.md",
    "docs/release-gating.md",
    "docs/release-runbook.md",
    "docs/tile-and-shape-coverage.md",
    "docs/v1-batch-plan.md",
]
REQUIRED_SCRIPTS = [
    "scripts/correctness_audit.py",
    "scripts/copy_docker_variant_artifacts.py",
    "scripts/prebuild_check.py",
    "scripts/release_build_plan.py",
    "scripts/run_built_artifact_benchmarks.py",
]
REQUIRED_FILES = [
    "README.md",
    "CARD.md",
    "VALIDATION.md",
    "build.toml",
    "flake.nix",
    "flake.lock",
    "benchmarks/RESULTS.md",
]
REQUIRED_DIRS = [
    "csrc",
    "torch-ext",
    "tests",
    "benchmarks",
    "examples",
]
ARTIFACT_NAMES = [
    "build",
    "dist",
    "result",
    "result-2",
]


def run(cmd: list[str], cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def warn(warnings: list[str], message: str) -> None:
    warnings.append(message)


def check_package(pkg: str, errors: list[str], warnings: list[str]) -> None:
    pkg_dir = ROOT / pkg
    if not pkg_dir.is_dir():
        fail(errors, f"{pkg}: package directory is missing")
        return

    for rel in REQUIRED_FILES:
        path = pkg_dir / rel
        if not path.is_file():
            fail(errors, f"{pkg}: missing {rel}")

    for rel in REQUIRED_DIRS:
        path = pkg_dir / rel
        if not path.is_dir():
            fail(errors, f"{pkg}: missing {rel}/")

    for name in ARTIFACT_NAMES:
        path = pkg_dir / name
        if path.exists() or path.is_symlink():
            fail(errors, f"{pkg}: remove build artifact {name}")

    for pycache in pkg_dir.rglob("__pycache__"):
        warn(warnings, f"{pkg}: ignored cache exists at {pycache.relative_to(ROOT)}")

    build_toml = pkg_dir / "build.toml"
    if not build_toml.is_file():
        return

    try:
        with build_toml.open("rb") as handle:
            config = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        fail(errors, f"{pkg}: invalid build.toml: {exc}")
        return

    name = config.get("general", {}).get("name")
    expected_names = {pkg, pkg.lower()}
    if name not in expected_names:
        fail(errors, f"{pkg}: general.name is {name!r}, expected one of {sorted(expected_names)!r}")

    source_entries: list[str] = []
    source_entries.extend(config.get("torch", {}).get("src", []))
    for kernel_name, kernel_cfg in config.get("kernel", {}).items():
        backend = kernel_cfg.get("backend")
        if backend != "cuda":
            fail(errors, f"{pkg}: kernel {kernel_name} backend is {backend!r}")
        source_entries.extend(kernel_cfg.get("src", []))

    for rel in source_entries:
        path = pkg_dir / rel
        if not path.is_file():
            fail(errors, f"{pkg}: build.toml references missing source {rel}")


def check_internal_dirs(errors: list[str]) -> None:
    tracked = run(["git", "ls-files", "internal-docs", "internal-tests"]).stdout.strip()
    if tracked:
        fail(errors, "internal-docs/ or internal-tests/ contains tracked files")


def check_docs(errors: list[str]) -> None:
    for rel in REQUIRED_DOCS:
        if not (ROOT / rel).is_file():
            fail(errors, f"missing required doc {rel}")

    for rel in REQUIRED_SCRIPTS:
        if not (ROOT / rel).is_file():
            fail(errors, f"missing required script {rel}")


def check_config(pkg: str, builder: str, errors: list[str]) -> None:
    result = run([builder, "check-config", "."], cwd=ROOT / pkg)
    if result.returncode != 0:
        detail = (result.stdout + result.stderr).strip()
        fail(errors, f"{pkg}: check-config failed\n{detail}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="also run kernel-builder-docker check-config for each package",
    )
    parser.add_argument(
        "--builder",
        default="/home/heima/suliang/PI/.hf-kernel-env/bin/kernel-builder-docker",
        help="kernel-builder-docker command path",
    )
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []

    for pkg in V1_PACKAGES + V2_CANDIDATE_PACKAGES:
        check_package(pkg, errors, warnings)
        if args.check_config:
            check_config(pkg, args.builder, errors)

    check_docs(errors)
    check_internal_dirs(errors)

    for item in warnings:
        print(f"WARN {item}")
    for item in errors:
        print(f"FAIL {item}")

    if errors:
        return 1

    print("prebuild check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
