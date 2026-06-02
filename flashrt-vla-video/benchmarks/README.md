# Benchmarks

Implemented benchmark groups:

- `benchmark_q_norm_rope.py`: decode-time Q/K RMSNorm + rotate-half RoPE with
  head_dim=128 and head counts 1, 4, 8, 16, 32, 48. The same file also covers
  packed-QKV split + Q/K RMSNorm + interleaved RoPE for video-token lengths
  64, 256, 1024, and 2520.

Planned benchmark groups:

- Patch embedding and bias/position fusion.
- Video tensor layout conversion and quantization.
- DiT/VAE helper kernels.
- FlashRT-real VLA/video shape families.
