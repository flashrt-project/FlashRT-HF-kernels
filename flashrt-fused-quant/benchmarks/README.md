# Benchmarks

Planned benchmark groups:

- Split and merged `SiLU(gate) * up + quant` against PyTorch eager chains.
- `residual + RMSNorm + quant` against separate PyTorch/CUDA launches.
- Decode, prefill, image-token, and video-token sequence lengths.
- Report both latency and effective memory bandwidth.
