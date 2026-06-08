# Source Sync

Synced from `official/FlashRT`:

- `csrc/kernels/norm.cu`: `ada_rms_norm_style` and `ada_rms_norm_style_fp8`
  math.
- `csrc/kernels/fusion.cu`: `gate_residual_ada_norm_fp8` fused residual,
  AdaRMSNorm, and FP8 output path.

Local adaptation:

- Public package/API names use generic adaptive-norm terminology.
- Only BF16 input and static FP8 output paths are included in this first public
  package version.
- Tensor-facing `torch.ops` bindings validate dtype, shape, contiguity, device,
  and even hidden dimensions.
- CUDA math uses explicit round-to-nearest add/multiply operations in key
  BF16-reference paths for stable public correctness behavior.
- FP8 validation gates on p99-zero and sparse boundary differences because
  PyTorch and CUDA FP8 casters can select adjacent FP8 grid points on rare ties.
