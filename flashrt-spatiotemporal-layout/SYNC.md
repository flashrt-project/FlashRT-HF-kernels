# Source Sync

Synced from FlashRT layout/cache idioms used in VLA, video, diffusion, and
world-model pipelines.

Local adaptation:

- Public package/API names use generic spatiotemporal and world-model
  terminology instead of model-specific names.
- Tensor-facing `torch.ops` bindings validate dtype, shape, contiguity, device,
  and BF16 layout contracts.
- The package intentionally exposes small layout/cache helpers. Larger serving
  orchestration, CUDA Graph capture, stream ownership, and pointer-level FlashRT
  APIs remain in the upstream FlashRT runtime.
