#!/usr/bin/env python3
"""Static correctness gate audit for the v1 batch.

Numerical proof is provided by scripts/accuracy_sweep.py. This script checks
that the repo documents that gate and has no known static correctness blocker.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DOCS = [
    "docs/correctness-gating.md",
    "docs/source-accuracy-results.md",
]


def main() -> int:
    for rel in REQUIRED_DOCS:
        doc = ROOT / rel
        if not doc.is_file():
            print(f"FAIL missing {rel}")
            return 1

    errors: list[str] = []
    evidence = (ROOT / "docs" / "source-accuracy-results.md").read_text()
    required_phrases = [
        "python scripts/accuracy_sweep.py --backend source --mode full --package all",
        "`flashrt-gemm-epilogues`",
        "`flashrt-vla-video`",
        "`flashrt-nvfp4`",
        "`flashrt-smallm-gemm`",
        "`flashrt-fused-quant`",
    ]
    for phrase in required_phrases:
        if phrase not in evidence:
            errors.append(f"docs/source-accuracy-results.md missing {phrase}")

    for error in errors:
        print(f"BLOCKER {error}")

    if errors:
        print("correctness audit failed: do not start release build")
        return 1

    print("correctness audit passed; run accuracy_sweep.py after any source change")
    return 0


if __name__ == "__main__":
    sys.exit(main())
