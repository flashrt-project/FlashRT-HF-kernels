#!/usr/bin/env python3
"""Checkpoint-backed OpenPI/PyTorch PI0.5 baseline benchmark.

This is the ecosystem baseline for the FlashRT HF runtime demo. It runs the
official OpenPI PyTorch model with real PI0.5 safetensors weights and records
steady-state `sample_actions` latency. It intentionally does not import or use
FlashRT.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any


IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _stats(xs: list[float]) -> dict[str, float]:
    ys = sorted(float(x) for x in xs)
    return {
        "n": float(len(ys)),
        "p50_ms": ys[int(0.50 * (len(ys) - 1))],
        "p90_ms": ys[int(0.90 * (len(ys) - 1))],
        "p95_ms": ys[int(0.95 * (len(ys) - 1))],
        "mean_ms": statistics.mean(ys),
        "min_ms": ys[0],
        "max_ms": ys[-1],
    }


def _resolve_weight_path(path: str) -> Path:
    p = Path(path)
    if p.is_dir():
        p = p / "model.safetensors"
    if not p.exists():
        raise FileNotFoundError(f"checkpoint weights not found: {p}")
    return p


def _maybe_repair_transformers_replace(openpi_root: Path) -> None:
    import transformers

    src = openpi_root / "openpi/models_pytorch/transformers_replace"
    if not src.exists():
        raise FileNotFoundError(f"transformers_replace source missing: {src}")
    dst = Path(transformers.__file__).resolve().parent
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def _check_transformers_replace(openpi_root: Path, repair: bool) -> dict[str, Any]:
    import transformers

    if repair:
        _maybe_repair_transformers_replace(openpi_root)

    try:
        from transformers.models.siglip import check

        replace_ok = bool(check.check_whether_transformers_replace_is_installed_correctly())
    except Exception:
        replace_ok = False

    return {
        "transformers_version": transformers.__version__,
        "transformers_path": str(Path(transformers.__file__).resolve().parent),
        "transformers_replace_ok": replace_ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openpi-root", required=True, help="Path to openpi src root")
    parser.add_argument("--checkpoint", required=True, help="PI0.5 checkpoint dir or model.safetensors")
    parser.add_argument("--prompt", default="pick up the red block and place it in the tray")
    parser.add_argument("--num-views", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--compile",
        choices=("off", "on"),
        default="off",
        help=(
            "OpenPI wraps sample_actions in torch.compile. Use off for a stable "
            "PyTorch eager baseline; use on only to diagnose the local Torch "
            "Inductor path."
        ),
    )
    parser.add_argument(
        "--repair-transformers-replace",
        action="store_true",
        help="Copy OpenPI transformers_replace files into the installed transformers package.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    openpi_root = Path(args.openpi_root).resolve()
    if str(openpi_root) not in sys.path:
        sys.path.insert(0, str(openpi_root))

    import numpy as np
    import torch
    import safetensors.torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    if args.compile == "off":
        # OpenPI assigns `self.sample_actions = torch.compile(...)` in the
        # constructor. For a strict eager baseline, make that assignment a no-op.
        torch.compile = lambda fn=None, *a, **k: fn if fn is not None else (lambda f: f)

    tr_info = _check_transformers_replace(openpi_root, args.repair_transformers_replace)
    if not tr_info["transformers_replace_ok"]:
        raise RuntimeError(
            "OpenPI transformers_replace is not installed correctly. Install "
            "transformers==4.53.2 and copy "
            f"{openpi_root}/openpi/models_pytorch/transformers_replace/* into "
            "the installed transformers package, or rerun with "
            "--repair-transformers-replace after installing transformers==4.53.2."
        )

    from openpi.models.model import Observation
    from openpi.models.pi0_config import Pi0Config
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    weight_path = _resolve_weight_path(args.checkpoint)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = Pi0Config(
        pi05=True,
        action_horizon=10,
        action_dim=32,
        dtype="bfloat16",
    )

    t0 = time.perf_counter()
    model = PI0Pytorch(cfg)
    safetensors.torch.load_model(model, str(weight_path), strict=False)
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model.to("cuda").eval()
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0

    tokenizer = PaligemmaTokenizer(max_len=cfg.max_token_len)
    state_np = np.zeros((cfg.action_dim,), dtype=np.float32)
    tokens_np, mask_np = tokenizer.tokenize(args.prompt, state_np)

    images = {}
    image_masks = {}
    for i, key in enumerate(IMAGE_KEYS):
        img = torch.randint(
            0, 256, (1, 3, 224, 224), device="cuda", dtype=torch.uint8
        ).float() / 127.5 - 1.0
        images[key] = img
        image_masks[key] = torch.full(
            (1,), i < args.num_views, device="cuda", dtype=torch.bool
        )

    obs = Observation(
        images=images,
        image_masks=image_masks,
        state=torch.from_numpy(state_np).to("cuda", torch.float32).view(1, -1),
        tokenized_prompt=torch.from_numpy(tokens_np).to("cuda", torch.long).view(1, -1),
        tokenized_prompt_mask=torch.from_numpy(mask_np).to("cuda", torch.bool).view(1, -1),
    )
    noise = torch.randn(
        (1, cfg.action_horizon, cfg.action_dim),
        device="cuda",
        dtype=torch.float32,
    )

    def call() -> torch.Tensor:
        with torch.no_grad():
            return model.sample_actions("cuda", obs, noise=noise, num_steps=args.steps)

    t0 = time.perf_counter()
    out = call()
    torch.cuda.synchronize()
    first_ms = (time.perf_counter() - t0) * 1000.0
    if not torch.isfinite(out).all():
        raise RuntimeError("OpenPI baseline output contains NaN or Inf")

    for _ in range(args.warmup):
        out = call()
        torch.cuda.synchronize()
        if not torch.isfinite(out).all():
            raise RuntimeError("OpenPI baseline output contains NaN or Inf during warmup")

    times_ms = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        out = call()
        torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    payload = {
        "name": "openpi_pytorch_baseline",
        "status": "pass",
        "mode": "torch_compile" if args.compile == "on" else "torch_eager",
        "checkpoint": str(weight_path),
        "openpi_root": str(openpi_root),
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "num_views": args.num_views,
        "steps": args.steps,
        "warmup": args.warmup,
        "iters": args.iters,
        "load_s": load_s,
        "first_ms": first_ms,
        "output_shape": list(out.shape),
        "output_finite": bool(torch.isfinite(out).all().item()),
        "transformers": tr_info,
        "latency": _stats(times_ms),
        "times_ms": times_ms,
        "note": (
            "Official OpenPI PyTorch PI0.5 model path with real safetensors "
            "weights. This is the ecosystem baseline; it does not use FlashRT."
        ),
    }

    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
