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

Native CUDA FlashRT grouped MoE GEMV kernels for dynamic device-side routing.
Version 2 covers BF16-activation/NVFP4-weight W4A16 and fully NVFP4 W4A4.

Available functions:

- `w4a16_decode_gemv_bf16`
- `grouped_w4a16_gemv_bf16`
- `quantize_activations_nvfp4_bf16`
- `quantize_weights_nvfp4_bf16`
- `grouped_w4a4_gemv_bf16`
- `grouped_w4a4_gemv_from_bf16`

W4A4 routing is token-major: packed activations `[M,K/2]`, device indices
`[M,top_k]`, packed expert library `[E,N,K/2]`, output `[M,top_k,N]`.
Use `M=routed_pairs, top_k=1` when each expert receives a distinct activation.
See the repository README for static-buffer and CUDA Graph usage.
