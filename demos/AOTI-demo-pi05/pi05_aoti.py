#!/usr/bin/env python3
"""AOTInductor route for pi05-fast.

``torch.export`` + AOTInductor compiles a model to a standalone ``.pt2`` that
loads without re-tuning -- the key to fast cold starts and to ZeroGPU Spaces,
where every request forks a fresh process and JIT ``torch.compile`` caches do
not carry over (see the HF "ZeroGPU + AOTI" guide).

pi05's 10-step denoise loop does not export cleanly as one graph: the sync-free
pass removes the per-step KV-cache copy (so it's a single ``torch.compile``
graph), but a dynamic prefix length and a fake-tensor attention-mask broadcast
still break ``torch.export``. So this AOTI route compiles the cleanly-exportable,
static-shape SigLIP vision embed -- which runs on every inference -- to a
``.pt2`` artifact, and keeps the denoise loop on ``torch.compile``. Extending
AOTI to the denoise loop (freeze the prefix length, resolve the mask) is future
work toward a ZeroGPU Space.

On a warm persistent process AOTI and compile reach the same speed; AOTI's win
is the portable artifact, not extra throughput.
"""

from __future__ import annotations

import types
from pathlib import Path

import torch

CACHE_DIR = Path.home() / ".cache" / "pi05-fast"


class _VisionEmbed(torch.nn.Module):
    """Wraps pi05's image embed (vision tower + projector) for export."""

    def __init__(self, paligemma_with_expert) -> None:
        super().__init__()
        self.pwe = paligemma_with_expert
        self.scale = paligemma_with_expert.paligemma.config.text_config.hidden_size ** 0.5

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.pwe.paligemma.model.get_image_features(image)
        return features.pooler_output * self.scale


def _example_image(policy, batch) -> torch.Tensor:
    for key in ("observation.images.image", "observation.images.image2"):
        if key in batch and torch.is_tensor(batch[key]):
            img = batch[key]
            img = img if img.dim() == 4 else img.unsqueeze(0)
            return img[:1].to(torch.float32)
    device = next(policy.parameters()).device
    return torch.zeros(1, 3, 224, 224, device=device, dtype=torch.float32)


def apply_export_aoti(policy, batch, *, cache_dir: Path = CACHE_DIR, serialize: bool = True) -> None:
    """Compile the SigLIP vision embed to a ``.pt2`` and compile the denoise loop."""
    from pi05_fast import apply_compile

    pwe = policy.model.paligemma_with_expert
    cache_dir.mkdir(parents=True, exist_ok=True)
    package_path = cache_dir / "vision_embed.pt2"

    if serialize and package_path.exists():
        loaded = torch._inductor.aoti_load_package(str(package_path))
    else:
        module = _VisionEmbed(pwe).eval()
        example = _example_image(policy, batch)
        with torch.inference_mode():
            exported = torch.export.export(module, (example,))
            path = torch._inductor.aoti_compile_and_package(
                exported,
                package_path=str(package_path),
                inductor_configs={"max_autotune": True},
            )
        loaded = torch._inductor.aoti_load_package(path)

    out_dtype = policy.model.dtype if hasattr(policy.model, "dtype") else torch.bfloat16

    def embed_image(self, image: torch.Tensor):
        features = loaded(image.to(torch.float32))
        return features.to(out_dtype) if features.dtype != out_dtype else features

    pwe.embed_image = types.MethodType(embed_image, pwe)
    apply_compile(policy)
