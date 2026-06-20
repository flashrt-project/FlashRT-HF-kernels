---
license: apache-2.0
tags:
- cuda
- flashrt
- attention
- fp8
- kv-cache
- transformers
---

# FlashRT FP8 KV Attention

Native CUDA XQA attention for BF16 queries over FP8 E4M3 paged K/V cache.

## Available Functions

- `xqa_bf16_fp8kv`
- `causal_spec_mask`
- `default_page_table`
- `allocate_workspace`

## Scope

v1 is a fixed-shape public package for the production Qwen3.6-style path:

- BF16 Q/O
- FP8 E4M3 K/V cache
- `24` Q heads, `4` KV heads, head dim `256`
- page size `128`
- speculative/decode `q_seq <= 32`

This is not a generic FlashAttention replacement. It is the direct FP8-KV XQA
decode/verify kernel used to keep long-context transformer runtimes off BF16
KV cache bandwidth.

## Minimal Usage

```python
from kernels import get_kernel

attn = get_kernel("flashrt/fp8-kv-attention", trust_remote_code=True)
out = attn.xqa_bf16_fp8kv(q_bf16, k_cache_fp8, v_cache_fp8)
```

Pass explicit `page_table`, `seq_lens`, `mask`, `out`, `semaphores`, and
`scratch` tensors for CUDA Graph/static-buffer runtimes.
