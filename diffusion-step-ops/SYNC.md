# Source Sync

This package implements standalone tensor-API versions of FlashRT diffusion
step helpers from:

- `official/FlashRT/csrc/kernels/elementwise.cu`
- `official/FlashRT/csrc/kernels/elementwise.cuh`

The Hub package intentionally avoids importing the full FlashRT `elementwise.cu`
translation unit, because that file contains many unrelated serving/runtime
helpers. This keeps the public package small and auditable.
