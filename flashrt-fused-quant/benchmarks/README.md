# Benchmarks

Current benchmark scope:

- Split and merged `SiLU(gate) * up + quant` against PyTorch eager chains.
- Decode, prefill, image-token, and video-token sequence lengths.
- Report both latency and effective memory bandwidth.

Queued benchmark groups for later source slices:

- `residual + RMSNorm + quant` against separate PyTorch/CUDA launches.
