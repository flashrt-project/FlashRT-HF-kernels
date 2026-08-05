---
tags: [kernel, cuda, attention, inference, cuda-graphs]
library_name: kernels
license: apache-2.0
---

# Masked MHA Runtime

Allocation-free FP16/BF16 attention that masks padded logits inside softmax,
removing the per-call `-inf` pre-fill. BF16 accepts fused-QKV token strides.

## API

- `forward(q, k, v, *, scale=None)`
- `forward_static(q, k, v, *, logits, out, scale=None)`
- `allocate_workspace(q, k)`

Inputs use `(sequence, heads, head_dim)`. `forward_static` is the CUDA Graph
hot-path API: allocate `logits` and `out` once and reuse their addresses.
Rows wider than 1024 keys use a deterministic multi-pass softmax.

This package contains the native masked-MHA execution path validated in
FlashRT's GROOT N1.7 Thor runtime. It is separate from FlashAttention-4.
