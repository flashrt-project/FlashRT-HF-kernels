# FlashRT FP8 Cross-Attention for Blackwell

Native CUTLASS FP8 E4M3 GQA self/cross-attention with BF16 output for SM100,
SM103, and SM110. It is non-causal and supports unequal query/KV lengths,
which complements `flashrt/fp8-prefill-attention-blackwell`'s SM120 causal
self-attention contract.

```python
from kernels import get_kernel

attn = get_kernel("flashrt/fp8-cross-attention-blackwell", version=1)
out = attn.fp8_gqa_cross_attention_bf16(
    q, k, v,
    query_scale=q_scale,
    key_scale=k_scale,
    value_scale=v_scale,
)
```

Q/K/V use contiguous `[B,S,H,128]` FP8 E4M3FN layout. Query heads must be
divisible by KV heads. Scales dequantize FP8 values (`real ~= fp8 * scale`).
Pass `output`, `lse`, and `workspace` buffers for an allocation-free CUDA Graph
hot path. The implementation never owns process-global CUDA buffers.

CUDA 13 is required. See `VALIDATION.md` before making a hardware claim.
