#!/usr/bin/env python3
"""Print or execute the v1 release-candidate build-window command sequence."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = "/home/heima/suliang/PI/.hf-kernel-env/bin/kernel-builder-docker"
PACKAGES = [
    "flashrt-gemm-epilogues",
    "flashrt-vla-video",
    "flashrt-nvfp4",
    "flashrt-smallm-gemm",
    "flashrt-fused-quant",
]
BUILD_VARIANTS = [
    "torch211-cxx11-cu128-x86_64-linux",
]


def command_plan(builder: str) -> list[tuple[str, list[str]]]:
    commands: list[tuple[str, list[str]]] = [
        (
            ".",
            [
                "python",
                "scripts/accuracy_sweep.py",
                "--backend",
                "source",
                "--mode",
                "full",
                "--package",
                "all",
                "--quiet",
            ],
        ),
        (".", ["python", "scripts/correctness_audit.py"]),
        (".", ["python", "scripts/prebuild_check.py", "--check-config"]),
    ]
    for pkg in PACKAGES:
        for variant in BUILD_VARIANTS:
            commands.append(
                (
                    pkg,
                    [
                        builder,
                        "build",
                        "--variant",
                        variant,
                        "--max-jobs",
                        "1",
                        "--cores",
                        "8",
                        ".",
                    ],
                )
            )
        commands.append((pkg, [builder, "check-builds", "."]))
    for variant in BUILD_VARIANTS:
        commands.append(
            (
                ".",
                [
                    "python",
                    "scripts/copy_docker_variant_artifacts.py",
                    "--variant",
                    variant,
                ],
            )
        )
    return commands


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--builder",
        default=BUILDER,
        help="kernel-builder-docker command path",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run the plan instead of printing it",
    )
    args = parser.parse_args()

    for cwd, cmd in command_plan(args.builder):
        print(f"$ cd {cwd} && {' '.join(cmd)}")
        if args.execute:
            result = subprocess.run(cmd, cwd=ROOT / cwd, check=False)
            if result.returncode != 0:
                return result.returncode
    if not args.execute:
        print("\nDry run only. Add --execute during the release-candidate build window.")
        print("This plan builds and copies the selected release-candidate variants.")
        print("Run kernel-builder build-and-copy separately for the full HF variant matrix.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
