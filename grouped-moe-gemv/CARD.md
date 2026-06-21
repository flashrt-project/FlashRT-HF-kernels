---
library_name: kernels
license: apache-2.0
tags:
  - cuda
  - native-cuda
  - flashrt
  - moe
  - nvfp4
  - blackwell
---

# grouped-moe-gemv

Native CUDA FlashRT grouped MoE GEMV kernels for BF16 activations and NVFP4
weights.

Available functions:

- `w4a16_decode_gemv_bf16`
- `grouped_w4a16_gemv_bf16`
