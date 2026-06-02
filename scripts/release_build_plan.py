#!/usr/bin/env python3
"""Print or execute the v1 full build-window command sequence."""

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


def command_plan(builder: str) -> list[tuple[str, list[str]]]:
    commands: list[tuple[str, list[str]]] = [
        (".", ["python", "scripts/prebuild_check.py", "--check-config"]),
    ]
    for pkg in PACKAGES:
        commands.append((pkg, [builder, "build", "."]))
        commands.append((pkg, [builder, "check-builds", "."]))
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
        print("\nDry run only. Add --execute during the release build window.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
