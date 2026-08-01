---
tags: [kernels, cuda, blackwell, fp8, attention, gqa]
library_name: kernels
---

# fp8-cross-attention-blackwell

## Available functions

- `fp8_gqa_cross_attention_bf16`

Non-causal FP8 GQA self/cross-attention for SM100/103/110, with BF16 output.

```python
fp8_gqa_cross_attention_bf16(
    query, key, value, *, query_scale, key_scale, value_scale,
    output=None, lse=None, workspace=None
) -> output
```

Q is contiguous FP8 E4M3FN `[B,Sq,Hq,128]`; K/V are contiguous FP8 E4M3FN
`[B,Sk,Hkv,128]`; `Hq % Hkv == 0`. Output is BF16 with Q's shape. The operation
is non-causal. Unequal and non-128-aligned `Sq/Sk` are supported. For a static
hot path, provide BF16 `output`, FP32 `lse [B,Hq,round_up(Sq,128)]`, and a
contiguous CUDA uint8 workspace. CUDA 13 and SM100/103/110 are required.
