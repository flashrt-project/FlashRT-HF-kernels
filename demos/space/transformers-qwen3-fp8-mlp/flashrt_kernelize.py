"""Register FlashRT kernel-layers through the official transformers/kernels mechanism.

A single `kernelize(model)` call swaps every decorated layer (e.g. RMSNorm, used at
100+ sites across the model zoo) for a FlashRT kernel-layer -- the same mechanism
gpt-oss uses to map RMSNorm to Liger kernels, here pointed at FlashRT (Blackwell).

The kernel-layer classes are bundled locally (``flashrt_layers/``) and call
``get_kernel`` for the published FlashRT ops, so no Hub layer-repo is required.
"""

from __future__ import annotations

from pathlib import Path

from kernels import LocalLayerRepository, Mode, kernelize, register_kernel_mapping

_REPO = Path(__file__).resolve().parent / "flashrt_layers"

FLASHRT_KERNEL_MAPPING = {
    "RMSNorm": {
        "cuda": {
            Mode.INFERENCE: LocalLayerRepository(
                _REPO, package_name="flashrt_layers", layer_name="RMSNorm"
            )
        }
    },
}


def register_flashrt_kernels(mapping: dict | None = None) -> None:
    """Register the FlashRT kernel mapping with the kernels library."""

    register_kernel_mapping(mapping or FLASHRT_KERNEL_MAPPING)


def kernelize_model(model, device: str = "cuda"):
    """Register FlashRT kernels and kernelize ``model`` for inference."""

    register_flashrt_kernels()
    return kernelize(model, mode=Mode.INFERENCE, device=device)
