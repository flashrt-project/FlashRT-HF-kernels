#!/usr/bin/env python3
"""Run the static-vs-dynamic FP8 reproduction suite."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNTIME = ROOT / "demos/runtime-demo"


def extract_json(text: str):
    match = re.search(r"\{\s*\"name\".*\}\s*$", text, re.S)
    if not match:
        return None
    return json.loads(match.group(0))


def run_cmd(label: str, cmd: list[str], out_dir: Path) -> dict:
    print(f"== {label} ==")
    print(" ".join(cmd))
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    out_path = out_dir / f"{label}.log"
    out_path.write_text(proc.stdout + proc.stderr)
    parsed = extract_json(proc.stdout + proc.stderr)
    if parsed is not None:
        (out_dir / f"{label}.json").write_text(json.dumps(parsed, indent=2) + "\n")
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr)[-3000:]
        raise RuntimeError(f"{label} failed with exit={proc.returncode}; see {out_path}\n{tail}")
    return {"label": label, "log": str(out_path), "json": parsed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("PI05_CHECKPOINT", str(ROOT.parent / "checkpoints/pi05_libero_pytorch")),
    )
    parser.add_argument("--out-dir", default=str(ROOT / "internal-tests/runtime-demo/static-vs-dynamic-fp8"))
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument(
        "--suite",
        choices=("all", "all-fp8-e2e", "decoder-ffn-e2e", "decoder-loop", "microbench"),
        default="all",
    )
    parser.add_argument("--microbench-warmup", type=int, default=20)
    parser.add_argument("--microbench-iters", type=int, default=100)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    summary = []

    if args.suite in {"all", "all-fp8-e2e"}:
        for mode in ("static", "dynamic"):
            summary.append(
                run_cmd(
                    f"all_fp8_e2e_{mode}",
                    [
                        py,
                        str(RUNTIME / "dyn_full_all_fp8.py"),
                        mode,
                        "--checkpoint",
                        args.checkpoint,
                        "--warmup",
                        str(args.warmup),
                        "--iters",
                        str(args.iters),
                    ],
                    out_dir,
                )
            )

    if args.suite in {"all", "decoder-ffn-e2e"}:
        for mode in ("static", "dynamic"):
            summary.append(
                run_cmd(
                    f"decoder_ffn_e2e_{mode}",
                    [
                        py,
                        str(RUNTIME / "dyn_full.py"),
                        mode,
                        "--checkpoint",
                        args.checkpoint,
                        "--warmup",
                        str(args.warmup),
                        "--iters",
                        str(args.iters),
                    ],
                    out_dir,
                )
            )

    if args.suite in {"all", "decoder-loop"}:
        summary.append(
            run_cmd(
                "decoder_loop",
                [
                    py,
                    str(RUNTIME / "sd_decoder.py"),
                    "--checkpoint",
                    args.checkpoint,
                    "--warmup",
                    str(args.warmup),
                    "--iters",
                    str(args.iters),
                ],
                out_dir,
            )
        )

    if args.suite in {"all", "microbench"}:
        summary.append(
            run_cmd(
                "geglu_microbench",
                [
                    py,
                    str(HERE / "geglu_static_dynamic_microbench.py"),
                    "--shapes",
                    "headline",
                    "--warmup",
                    str(args.microbench_warmup),
                    "--iters",
                    str(args.microbench_iters),
                ],
                out_dir,
            )
        )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
