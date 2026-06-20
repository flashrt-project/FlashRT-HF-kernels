---
license: apache-2.0
tags:
  - kernels
  - cuda
  - flashrt
  - transformers
  - vision-language
  - qwen3-vl
---

# flashrt/vl-transformer-primitives

FlashRT CUDA kernels for VL transformer post-processing and vision-token
layout helpers.

## Functions

```python
from kernels import get_kernel

vl = get_kernel("flashrt/vl-transformer-primitives")

vl.qwen3_q_norm_rope_qstage_bf16(q_pre, q_norm_weight, cos, sin)
vl.qwen3_k_norm_rope_kvwrite_bf16(k_pre, v_pre, k_norm_weight, cos, sin)
vl.qwen3_k_norm_rope_kvwrite_devpos_bf16(
    k_pre, v_pre, k_norm_weight, cos, sin, cur_pos, k_cache, v_cache
)
vl.avg_pool_vision_tokens_bf16(tokens, nv=2, h=16, w=16, pool_factor=2)
```

## Scope

- Qwen3/Qwen3-VL decode hot path: head dimension 128, full RoPE, BF16 tensors.
- Cosmos3/Qwen3-VL style vision token average pooling.
- Static-buffer and CUDA Graph friendly APIs.

This package does not include generic FlashAttention. HF already has dedicated
FlashAttention kernel packages; this package focuses on adjacent transformer
primitives that are commonly left as eager PyTorch glue.
