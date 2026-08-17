#!/usr/bin/env python3
"""Audit latest Kernel Hub and legacy-main build variant coverage."""

from __future__ import annotations

import argparse
import os
import tomllib
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_VARIANTS = (
    "torch211-cxx11-cu128-x86_64-linux",
    "torch211-cxx11-cu130-x86_64-linux",
    "torch212-cxx11-cu130-x86_64-linux",
    "torch212-cxx11-cu132-x86_64-linux",
    "torch213-cxx11-cu130-x86_64-linux",
    "torch213-cxx11-cu132-x86_64-linux",
)


def package_metadata(
    root: Path, package: str
) -> tuple[str, int, tuple[int, int], tuple[int, int] | None, bool]:
    path = root / package / "build.toml"
    if not path.is_file():
        raise RuntimeError(f"missing package build.toml: {path}")
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    general = config["general"]
    repo_id = general.get("hub", {}).get("repo-id", f"flashrt/{package}")
    cuda = general.get("cuda", {})
    minver = tuple(int(part) for part in str(cuda.get("minver", "0.0")).split("."))
    maxver_text = cuda.get("maxver")
    maxver = (
        tuple(int(part) for part in str(maxver_text).split("."))
        if maxver_text is not None
        else None
    )
    return repo_id, int(general["version"]), minver, maxver, "torch-noarch" in config


def variant_cuda_version(variant: str) -> tuple[int, int]:
    marker = next(part for part in variant.split("-") if part.startswith("cu"))
    digits = marker[2:]
    return int(digits[:-1]), int(digits[-1])


def build_variants(files: list[str]) -> set[str]:
    variants: set[str] = set()
    for path in files:
        parts = path.split("/")
        if len(parts) == 3 and parts[0] == "build" and parts[2] == "metadata.json":
            variants.add(parts[1])
    return variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packages", nargs="+", help="Local package directory names")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--variant",
        action="append",
        dest="variants",
        help="Required variant; repeat as needed (defaults to maintained x86 matrix)",
    )
    parser.add_argument(
        "--skip-legacy-main",
        action="store_true",
        help="Do not require the old-client model main alias to match the latest kernel ref",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.repo_root).resolve()
    required = set(args.variants or DEFAULT_VARIANTS)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    api = HfApi(token=token)
    failures: list[str] = []

    for package in args.packages:
        repo_id, version, minver, maxver, noarch = package_metadata(root, package)
        revision = f"v{version}"
        if noarch:
            package_required = {"torch-universal"}
        else:
            package_required = {
                variant
                for variant in required
                if variant_cuda_version(variant) >= minver
                and (maxver is None or variant_cuda_version(variant) <= maxver)
            }
        kernel_files = api.list_repo_files(repo_id, repo_type="kernel", revision=revision)
        kernel_variants = build_variants(kernel_files)
        missing = sorted(package_required - kernel_variants)
        print(f"{repo_id}@{revision}: {len(kernel_variants)} variants")
        if missing:
            failures.append(f"{repo_id}@{revision} missing: {', '.join(missing)}")

        if args.skip_legacy_main:
            continue
        model_files = api.list_repo_files(repo_id, repo_type="model", revision="main")
        model_variants = build_variants(model_files)
        if model_variants != kernel_variants:
            only_kernel = sorted(kernel_variants - model_variants)
            only_model = sorted(model_variants - kernel_variants)
            failures.append(
                f"{repo_id} legacy main mismatch: only-kernel={only_kernel}, "
                f"only-model={only_model}"
            )

    if failures:
        raise SystemExit("Hub variant audit failed:\n- " + "\n- ".join(failures))
    print("Hub variant audit passed")


if __name__ == "__main__":
    main()
