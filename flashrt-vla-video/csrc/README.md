# CUDA Sources

Synced source slices:

- `q_norm_rope_bf16.cu`
- `q_norm_rope_bf16.cuh`

These implement generic BF16 per-row RMSNorm plus rotate-half RoPE for
head_dim=128 decode paths. The K variant also copies V into a cache-shaped
output.

The same source file also implements `qkv_split_norm_rope_bf16`, which reads
packed BF16 QKV with shape `(B, L, 3 * heads * head_dim)` and writes Q/K outputs
with shape `(B, L, heads, head_dim)` after Q/K RMSNorm and interleaved RoPE.
