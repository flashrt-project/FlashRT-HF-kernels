# CUDA Sources

Synced source slices:

- `q_norm_rope_bf16.cu`
- `q_norm_rope_bf16.cuh`

These implement generic BF16 per-row RMSNorm plus rotate-half RoPE for
head_dim=128 decode paths. The K variant also copies V into a cache-shaped
output.
