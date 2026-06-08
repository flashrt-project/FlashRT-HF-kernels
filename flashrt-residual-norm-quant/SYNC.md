# Source Sync

Synced from `official/FlashRT`:

- `csrc/kernels/norm.cu`: BF16 RMSNorm, RMSNorm to FP8, and residual-add
  RMSNorm to FP8 logic.
- `csrc/kernels/common.cuh`: packed BF16 conversion and block reduction
  helpers.

Local adaptation:

- Raw pointer APIs were converted to Tensor-based `torch.ops` bindings.
- Only BF16 + affine weight + static FP8 scale paths are included in this
  package version.
- AdaRMSNorm, noweight, FP16, INT8, and NVFP4 variants are intentionally left
  for follow-up packages or versions.
