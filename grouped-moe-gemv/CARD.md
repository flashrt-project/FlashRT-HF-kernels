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

Load version 2 with both current and legacy `kernels` clients:

```python
from kernels import get_kernel

try:
    moe = get_kernel(
        "flashrt/grouped-moe-gemv", version=2, trust_remote_code=True
    )
except TypeError:  # kernels==0.12.x
    moe = get_kernel("flashrt/grouped-moe-gemv", version=2)
```

Use `quantize_activations_nvfp4_bf16` once for `[M,K]`, followed by
`grouped_w4a4_gemv_bf16` for all `[M,top_k]` routes. The convenience function
`grouped_w4a4_gemv_from_bf16` performs both calls and accepts preallocated
`packed`, `sfa`, and `out` buffers for CUDA Graph capture.
