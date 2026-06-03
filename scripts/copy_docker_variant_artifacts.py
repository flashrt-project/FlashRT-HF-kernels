#!/usr/bin/env python3
"""Copy one built kernel-builder variant from the Docker Nix store to build/."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = [
    "flashrt-gemm-epilogues",
    "flashrt-fp8-ffn",
    "flashrt-vla-video",
    "flashrt-nvfp4",
    "flashrt-smallm-gemm",
    "flashrt-fused-quant",
]


def run(cmd: list[str], *, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def copy_variant(container: str, package: str, variant: str) -> tuple[bool, str]:
    result = ROOT / package / "result"
    if not result.is_symlink():
        return False, f"{package}: missing result symlink; run kernel-builder build first"

    store_path = result.readlink()
    if not str(store_path).startswith("/nix/store/"):
        return False, f"{package}: result target is not a Nix store path: {store_path}"

    dst = f"/work/FlashRT-HF-kernels/{package}/build"
    src = f"{store_path}/{variant}"
    script = (
        "set -e\n"
        f"test -d {src!r}\n"
        f"mkdir -p {dst!r}\n"
        f"rm -rf {dst!r}/{variant!r}\n"
        f"cp -r {src!r} {dst!r}/\n"
        f"chmod -R +w {dst!r}\n"
    )
    proc = run(["docker", "exec", container, "sh", "-lc", script])
    if proc.returncode != 0:
        detail = (proc.stdout + proc.stderr).strip()
        return False, f"{package}: copy failed from {src}\n{detail}"
    return True, f"{package}: copied {variant}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="torch211-cxx11-cu128-x86_64-linux")
    parser.add_argument("--container", default="flashrt-hf-kernel-env")
    parser.add_argument("--package", default="all", help="all or comma-separated package names")
    args = parser.parse_args()

    packages = PACKAGES if args.package == "all" else [p.strip() for p in args.package.split(",") if p.strip()]
    failed = False
    for package in packages:
        ok, message = copy_variant(args.container, package, args.variant)
        print(("OK " if ok else "FAIL ") + message)
        failed = failed or not ok
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
