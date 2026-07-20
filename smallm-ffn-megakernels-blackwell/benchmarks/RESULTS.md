# RTX 5090 source gate

CUDA 13.0 and a PyTorch 2.11 development environment. All FlashRT calls reuse
static output/scratch buffers. References execute the exact dequantized FP8
math and documented BF16/FP8 rounding points. Values are microseconds.

| Region | M | FlashRT | eager reference | compiled reference | vs compile |
|---|---:|---:|---:|---:|---:|
| gated residual 1024/4096 | 8 | 10.27 | 94.04 | 57.15 | 5.57x |
| gated residual 1024/4096 | 21 | 10.27 | 90.62 | 60.36 | 5.88x |
| gated residual 1024/4096 | 32 | 12.31 | 96.30 | 59.72 | 4.85x |
| residual 512/2048 | 51 | 10.27 | 71.91 | 56.67 | 5.52x |
| residual 512/2048 | 144 | 10.27 | 87.05 | 61.09 | 5.95x |
| split residual 512/2048 | 188 | 14.41 | 100.25 | 64.13 | 4.45x |

This is a source acceptance table, not the final Hub claim. HF Jobs artifacts
must reproduce correctness and original-source latency before publication.
