---
license: apache-2.0
tags:
- cuda
- flashrt
- transformers
- linear-attention
- bf16
---

# FlashRT Linear Attention Primitives

This package contains CUDA kernels for BF16 small-M linear projections and
linear-attention layout/gating helpers. It is designed for transformer decode
and verify hot paths where a full PyTorch op sequence creates excessive launch
and memory traffic.

## Available Functions

- `bf16_matvec`
- `bf16_smallm_matmul`
- `split_qkv_broadcast_bf16`
- `partial_rope_qk_bf16`
- `gated_delta_prepare_bf16`

## Usage

```python
from kernels import get_kernel

lap = get_kernel("flashrt/linear-attention-primitives")
out = lap.bf16_matvec(x, weight)
q, k, v = lap.split_qkv_broadcast_bf16(packed, 16, 16, 48, 128)
```

The APIs are Tensor APIs, not FlashRT serving-internal pointer APIs. They can
also be called with preallocated output tensors for static-buffer runtimes.

## Scope

The first release covers the strict source-validated subset used by FlashRT
runtime experiments. It does not package generic FlashAttention, which is
already available in the Hugging Face kernels ecosystem.
