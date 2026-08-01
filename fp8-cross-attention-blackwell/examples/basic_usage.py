from kernels import get_kernel

attention = get_kernel("flashrt/fp8-cross-attention-blackwell", version=1)
output = attention.fp8_gqa_cross_attention_bf16(
    query, key, value,
    query_scale=query_scale,
    key_scale=key_scale,
    value_scale=value_scale,
)
