# Benchmarks

Current benchmark scope:

- Split and merged `SiLU(gate) * up + quant` against PyTorch eager chains.
- Decode, prefill, image-token, and video-token sequence lengths.
- Report both latency and effective memory bandwidth.

The public HF-style benchmark script currently covers split and merged kernel
latency over the v1 shape grid:

```text
benchmark_silu_mul_quant_nvfp4.py
```

Correctness is covered by package tests because the HF benchmark runner only
verifies one tensor output, while this kernel writes packed data and swizzled
scale-factor bytes.

Queued benchmark groups for later source slices:

- `residual + RMSNorm + quant` against separate PyTorch/CUDA launches.
