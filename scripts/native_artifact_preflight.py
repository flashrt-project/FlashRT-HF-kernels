#!/usr/bin/env python3
"""Fail-fast checks for a locally built native Kernel Hub artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


STRONG_TYPES = set("TtBbDdRrSs")


def run(command: list[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stdout}{result.stderr}"
        )
    return result.stdout


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_extension(artifact: Path) -> Path:
    extensions = sorted(artifact.glob("*.so"))
    if len(extensions) != 1:
        raise RuntimeError(
            f"expected exactly one extension in {artifact}, found {extensions}"
        )
    return extensions[0]


def check_symbols(extension: Path, patterns: list[str]) -> dict[str, str]:
    lines = run(["nm", "-D", "-C", str(extension)]).splitlines()
    matched: dict[str, str] = {}
    for pattern in patterns:
        regex = re.compile(pattern)
        candidates = [line for line in lines if regex.search(line)]
        strong = []
        for line in candidates:
            fields = line.split(maxsplit=2)
            if len(fields) >= 3 and fields[1] in STRONG_TYPES:
                strong.append(line)
        if not strong:
            detail = "\n".join(candidates) if candidates else "<no matching symbol>"
            raise RuntimeError(
                f"required strong symbol /{pattern}/ is missing from {extension}\n{detail}"
            )
        matched[pattern] = strong[0]
    return matched


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--package", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--defined-symbol", action="append", default=[])
    parser.add_argument("--cubin-pattern", action="append", default=[])
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()

    artifact = args.artifact.resolve()
    metadata = artifact / "metadata.json"
    if not metadata.is_file():
        raise RuntimeError(f"missing metadata.json in {artifact}")
    if artifact.name != args.variant:
        raise RuntimeError(
            f"artifact directory {artifact.name!r} does not match variant {args.variant!r}"
        )
    metadata_value = json.loads(metadata.read_text(encoding="utf-8"))
    expected_metadata = {
        "name": args.package,
        "version": args.version,
    }
    for key, expected in expected_metadata.items():
        if metadata_value.get(key) != expected:
            raise RuntimeError(
                f"metadata {key}={metadata_value.get(key)!r}, expected {expected!r}"
            )
    backend = metadata_value.get("backend")
    if not isinstance(backend, dict) or backend.get("type") != args.backend:
        raise RuntimeError(
            f"metadata backend={backend!r}, expected type {args.backend!r}"
        )
    extension = find_extension(artifact)
    symbols = check_symbols(extension, args.defined_symbol)
    cubins = run(["cuobjdump", "--list-elf", str(extension)])
    for pattern in args.cubin_pattern:
        if re.search(pattern, cubins) is None:
            raise RuntimeError(f"required cubin /{pattern}/ is missing from {extension}")

    source_sha = run(["git", "rev-parse", "HEAD"]).strip()
    receipt = {
        "package": args.package,
        "variant": args.variant,
        "source_sha": source_sha,
        "artifact": str(artifact),
        "extension": extension.name,
        "extension_sha256": sha256(extension),
        "metadata_sha256": sha256(metadata),
        "metadata": {
            "name": metadata_value["name"],
            "version": metadata_value["version"],
            "backend": backend,
        },
        "strong_symbols": symbols,
        "cubin_patterns": args.cubin_pattern,
        "status": "pass",
    }
    rendered = json.dumps(receipt, indent=2, sort_keys=True)
    print(rendered)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, OSError) as error:
        print(f"FAIL {error}", file=sys.stderr)
        sys.exit(1)
