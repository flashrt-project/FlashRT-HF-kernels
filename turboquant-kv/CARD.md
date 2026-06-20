---
license: apache-2.0
tags:
  - kernels
  - cuda
  - flashrt
  - transformers
  - kv-cache
  - quantization
---

# flashrt/turboquant-kv

FlashRT CUDA kernels for TurboQuant-style KV cache unpack and K/V combine.

## Functions

```python
from kernels import get_kernel

tq = get_kernel("flashrt/turboquant-kv")

tq.unpack_packed_bf16(k_idx_packed, k_qjl_packed, v_idx_packed, cb_k_mse, cb_v, 3, 4)
tq.unpack_packed_mixed(k_idx_packed, k_qjl_packed, v_idx_packed, cb_k_mse, cb_v, 3, 4)
tq.combine_kv_bf16(k_mse, k_qjl, v_unit, k_norm, k_rnorm, v_norm, coef=0.125)
```

This package is a building block for cache-compression demos. It does not
claim to be a complete KV-cache runtime by itself.
