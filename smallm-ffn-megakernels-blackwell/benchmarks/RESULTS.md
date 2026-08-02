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

## NVIDIA Thor SM110 installed artifact

PyTorch 2.11.0+cu130, CUDA 13.0, static buffers. Full 10-case correctness,
`torch.compile(fullgraph=True)`, and CUDA Graph checks pass.

| Region | M | FlashRT us | PyTorch reference us | vs reference |
|---|---:|---:|---:|---:|
| gated residual 1024/4096 | 1 | 27.19 | 294.12 | 10.82x |
| gated residual 1024/4096 | 8 | 27.00 | 389.73 | 14.44x |
| gated residual 1024/4096 | 21 | 45.47 | 522.26 | 11.49x |
| gated residual 1024/4096 | 32 | 45.22 | 557.98 | 12.34x |
| residual 512/2048 | 1 | 22.46 | 131.52 | 5.85x |
| residual 512/2048 | 51 | 33.78 | 246.47 | 7.30x |
| residual 512/2048 | 144 | 57.08 | 354.68 | 6.21x |
| residual 512/2048 | 188 | 72.57 | 382.29 | 5.27x |

The PyTorch column is the exact dequantized FP8 operation sequence, not a
FlashRT-native baseline. Public model claims must use the model pipeline gate.
