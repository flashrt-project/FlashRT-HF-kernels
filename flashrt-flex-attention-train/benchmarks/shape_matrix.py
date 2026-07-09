#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "benchmark.py"


DEFAULT_SHAPES = [
    {"name": "pi052_b4_p700_k5_c50", "batch": 4, "prefix_len": 700, "action_blocks": 5, "action_block_size": 50},
    {"name": "pi052_b2_p700_k5_c50", "batch": 2, "prefix_len": 700, "action_blocks": 5, "action_block_size": 50},
    {"name": "pi052_b1_p700_k5_c50", "batch": 1, "prefix_len": 700, "action_blocks": 5, "action_block_size": 50},
    {"name": "pi052_b4_p512_k5_c50", "batch": 4, "prefix_len": 512, "action_blocks": 5, "action_block_size": 50},
    {"name": "pi052_b4_p896_k5_c50", "batch": 4, "prefix_len": 896, "action_blocks": 5, "action_block_size": 50},
    {"name": "pi052_b4_p700_k1_c50", "batch": 4, "prefix_len": 700, "action_blocks": 1, "action_block_size": 50},
    {"name": "pi052_b4_p700_k8_c50", "batch": 4, "prefix_len": 700, "action_blocks": 8, "action_block_size": 50},
]


def parse_presets(text: str) -> list[str]:
    if text == "a100":
        return [
            "default",
            "torch_default_explicit",
            "a100_d256_bwd_32x64",
            "a100_d256_bwd_32x128",
            "a100_d256_bwd_64x64",
            "a100_d256_contig_safe",
            "a100_d256_contig_prescale",
            "a100_d256_contig_write_dq_false",
        ]
    if text == "consumer":
        return [
            "default",
            "torch_default_explicit",
            "a100_d256_bwd_32x64",
            "a100_d256_bwd_64x128",
            "a100_d256_contig_safe",
            "a100_d256_contig_prescale",
        ]
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_block_sizes(text: str) -> list[tuple[int, int]]:
    if text == "default":
        return [(128, 128)]
    if text == "a100":
        return [(64, 64), (64, 128), (128, 64), (128, 128)]
    if text == "full":
        return [(q, kv) for q in (16, 32, 64, 128) for kv in (32, 64, 128)]
    out = []
    for item in text.split(","):
        item = item.strip().lower()
        if not item:
            continue
        q, kv = item.split("x", 1)
        out.append((int(q), int(kv)))
    return out


def load_shapes(path: str | None) -> list[dict]:
    if path is None:
        return DEFAULT_SHAPES
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("shape file must be a JSON list")
    return data


def extract_json(stdout: str) -> dict:
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start < 0 or end < start:
        return {"raw_stdout": stdout}
    return json.loads(stdout[start : end + 1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--mode", choices=["fwd", "fwdbwd", "all"], default="all")
    parser.add_argument("--backend", default="torch-flex", help="comma-separated subset of {package, torch-flex, manual} or 'all'")
    parser.add_argument("--presets", default="consumer", help="'consumer', 'a100', or comma-separated preset names")
    parser.add_argument("--block-mask-sizes", default="default", help="'default', 'a100', 'full', or comma-separated QxKV sizes")
    parser.add_argument("--shapes-json")
    parser.add_argument("--output", default=str(ROOT / "benchmarks" / "matrix_results.jsonl"))
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    shapes = load_shapes(args.shapes_json)
    presets = parse_presets(args.presets)
    block_sizes = parse_block_sizes(args.block_mask_sizes)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")

    for shape in shapes:
        for preset in presets:
            for block_q, block_kv in block_sizes:
                cmd = [
                    sys.executable,
                    str(BENCH),
                    "--device",
                    args.device,
                    "--dtype",
                    args.dtype,
                    "--batch",
                    str(shape["batch"]),
                    "--heads",
                    str(args.heads),
                    "--kv-heads",
                    str(args.kv_heads if args.kv_heads is not None else args.heads),
                    "--head-dim",
                    str(args.head_dim),
                    "--prefix-len",
                    str(shape["prefix_len"]),
                    "--action-blocks",
                    str(shape["action_blocks"]),
                    "--action-block-size",
                    str(shape["action_block_size"]),
                    "--warmup",
                    str(args.warmup),
                    "--iters",
                    str(args.iters),
                    "--mode",
                    args.mode,
                    "--backend",
                    args.backend,
                    "--flex-preset",
                    preset,
                    "--block-mask-q",
                    str(block_q),
                    "--block-mask-kv",
                    str(block_kv),
                ]
                proc = subprocess.run(cmd, cwd=ROOT.parent, text=True, capture_output=True)
                row = {
                    "shape_name": shape.get("name", ""),
                    "preset": preset,
                    "block_mask": f"{block_q}x{block_kv}",
                    "returncode": proc.returncode,
                }
                if proc.returncode == 0:
                    row.update(extract_json(proc.stdout))
                else:
                    row["stdout"] = proc.stdout[-4000:]
                    row["stderr"] = proc.stderr[-4000:]
                with output.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, sort_keys=True) + "\n")
                print(json.dumps(row, sort_keys=True))
                if proc.returncode != 0 and args.fail_fast:
                    return proc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
