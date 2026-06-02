#!/usr/bin/env python3
"""Static correctness gate audit for the v1 batch.

This script does not prove numerical correctness by itself. It prevents us from
starting a release build while known correctness gaps are still documented.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

BLOCKERS = {
    "flashrt-vla-video": [
        "QKV split + norm + RoPE needs Q and K accuracy sweep across token/head grids.",
    ],
    "flashrt-smallm-gemm": [
        "Full K/N decode grid and random/dequantized references are not complete.",
    ],
    "flashrt-fused-quant": [
        "Full v1 shape-grid byte parity for packed outputs and scales is not complete.",
    ],
    "flashrt-nvfp4": [
        "Full v1 layout benchmark grid must be verified before release results.",
    ],
    "flashrt-gemm-epilogues": [
        "BF16 GEMM epilogue wrappers use loose tolerances and are not headline-ready.",
    ],
}


def main() -> int:
    doc = ROOT / "docs" / "correctness-gating.md"
    if not doc.is_file():
        print("FAIL missing docs/correctness-gating.md")
        return 1

    errors: list[str] = []
    for package, blockers in BLOCKERS.items():
        for blocker in blockers:
            errors.append(f"{package}: {blocker}")

    for error in errors:
        print(f"BLOCKER {error}")

    if errors:
        print("correctness audit failed: do not start release build")
        return 1

    print("correctness audit passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
