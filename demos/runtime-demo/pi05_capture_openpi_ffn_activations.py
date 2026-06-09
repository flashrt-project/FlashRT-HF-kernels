#!/usr/bin/env python3
"""Capture real PI0.5 FFN activations from the official OpenPI PyTorch model.

Run this inside the isolated OpenPI baseline venv. It does not import FlashRT
or HF Kernel Hub packages. The output `.pt` file is consumed by
`pi05_real_weight_swiglu.py` in a separate HF-kernel environment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _resolve_weight_path(path: str) -> Path:
    p = Path(path)
    if p.is_dir():
        p = p / "model.safetensors"
    if not p.exists():
        raise FileNotFoundError(f"checkpoint weights not found: {p}")
    return p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openpi-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--family", choices=("encoder", "decoder"), default="decoder")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--num-views", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--prompt", default="pick up the red block and place it in the tray")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    openpi_root = Path(args.openpi_root).resolve()
    if str(openpi_root) not in sys.path:
        sys.path.insert(0, str(openpi_root))

    import numpy as np
    import torch
    import safetensors.torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    # Hooks and torch.compile are a bad mix here. The capture path is a
    # correctness/calibration path, not a performance path.
    torch.compile = lambda fn=None, *a, **k: fn if fn is not None else (lambda f: f)

    from transformers.models.siglip import check

    if not check.check_whether_transformers_replace_is_installed_correctly():
        raise RuntimeError("OpenPI transformers_replace is not installed correctly")

    from openpi.models.model import Observation
    from openpi.models.pi0_config import Pi0Config
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = Pi0Config(
        pi05=True,
        action_horizon=10,
        action_dim=32,
        dtype="bfloat16",
    )
    model = PI0Pytorch(cfg)
    safetensors.torch.load_model(model, str(_resolve_weight_path(args.checkpoint)), strict=False)
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model.to("cuda").eval()

    if args.family == "encoder":
        layer = model.paligemma_with_expert.paligemma.language_model.layers[args.layer]
    else:
        layer = model.paligemma_with_expert.gemma_expert.model.layers[args.layer]

    captured = []

    def _pre_hook(_module, inputs):
        x = inputs[0].detach()
        x = x.reshape(-1, x.shape[-1]).to(torch.bfloat16).cpu()
        captured.append(x)

    handle = layer.mlp.register_forward_pre_hook(_pre_hook)

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

    with torch.no_grad():
        out = model.sample_actions("cuda", obs, noise=noise, num_steps=args.steps)
    torch.cuda.synchronize()
    handle.remove()

    if not captured:
        raise RuntimeError("activation hook did not capture any tensors")
    acts = torch.cat(captured, dim=0)
    if args.max_rows > 0:
        acts = acts[: args.max_rows].contiguous()

    payload = {
        "activations": acts,
        "metadata": {
            "family": args.family,
            "layer": args.layer,
            "shape": list(acts.shape),
            "dtype": str(acts.dtype),
            "source": "OpenPI PI0.5 PyTorch layer.mlp forward_pre_hook",
            "checkpoint": str(_resolve_weight_path(args.checkpoint)),
            "num_views": args.num_views,
            "steps": args.steps,
            "output_shape": list(out.shape),
            "output_finite": bool(torch.isfinite(out).all().item()),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps(payload["metadata"], indent=2))


if __name__ == "__main__":
    main()
