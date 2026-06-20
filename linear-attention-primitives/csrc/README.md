# CUDA Sources

- `linear_attention_primitives.cu`: BF16 small-M linear, QKV split/broadcast,
  partial RoPE, and Gated DeltaNet prepare kernels.
- `linear_attention_primitives.cuh`: exported C++ CUDA entry points.
