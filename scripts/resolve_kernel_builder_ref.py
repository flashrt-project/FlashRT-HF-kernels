#!/usr/bin/env python3
"""Resolve the builder revision that must parse and build a package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TORCH213_UPSTREAM_REV = "b5443818ffe69a740095bb1174ca55e7fbb34c00"


def resolve(package: Path, variant: str | None) -> str:
    lock_path = package / "flake.lock"
    lock = json.loads(lock_path.read_text())
    node = lock.get("nodes", {}).get("kernel-builder", {})
    locked = node.get("locked", {})
    if locked.get("type") != "github":
        raise RuntimeError(f"{lock_path}: kernel-builder must be a locked GitHub input")
    owner = locked.get("owner")
    repo = locked.get("repo")
    revision = locked.get("rev")
    if not all((owner, repo, revision)):
        raise RuntimeError(f"{lock_path}: incomplete locked kernel-builder input")

    # Fork-owned builder pins carry package-specific dependency schemas such
    # as CUTLASS 4.4/4.5 and their own variant matrices. Replacing one with a
    # generic upstream CLI makes check-config and the build disagree.
    if owner != "huggingface":
        pass
    elif variant and variant.startswith("torch213-"):
        owner, repo, revision = "huggingface", "kernels", TORCH213_UPSTREAM_REV
    return f"github:{owner}/{repo}/{revision}#kernel-builder"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--variant")
    args = parser.parse_args()
    print(resolve(args.package.resolve(), args.variant))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
