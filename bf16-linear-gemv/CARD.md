---
library_name: kernels
license: apache-2.0
tags:
  - cuda
  - native-cuda
  - flashrt
  - transformer
  - gemv
  - bf16
---

# bf16-linear-gemv

Native CUDA BF16 M=1 decode GEMV kernels from FlashRT.

Available functions:

- `bf16_decode_gemv_bf16`
- `bf16_decode_gemv_unrolled_bf16`

See `README.md` and `VALIDATION.md`.
