# Benchmarks

Implemented benchmark groups:

- `benchmark_q_norm_rope.py`: decode-time Q/K RMSNorm + rotate-half RoPE with
  head_dim=128 and head counts 1, 4, 8, 16, 32, 48. The same file also covers
  packed-QKV split + Q/K RMSNorm + interleaved RoPE for video-token lengths
  1, 4, 16, 64, 256, 1024, 2520, and 4096.
- Internal tile sweeps cover `FLASHRT_QKV_ROPE_BLOCK_SIZE in {128, 256, 512}`
  before changing the package default.

Planned benchmark groups:

- Patch embedding and bias/position fusion.
- Video tensor layout conversion and quantization.
- DiT/VAE helper kernels.
- FlashRT-real VLA/video shape families.

## Comparison Stack

Public results for this package should follow
`../../docs/kernel-comparison-matrix.md`.

- Q/K RMSNorm + RoPE and packed-QKV split + Q/K norm + RoPE compare against
  PyTorch eager and `torch.compile` on the full token/head grid.
- If separate FlashRT or CUDA launch-chain baselines exist, report them as the
  unfused strong path to isolate the value of fusion.
- Attention and model-block demos must hold the attention backend constant on
  both paths. The claimed speedup is the postprocess or sublayer delta, not a
  change of attention algorithm.
- Wan/VLA/video examples are promoted only after both microkernel and
  model-block results are reported.
