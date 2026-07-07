# Results

Source-extension benchmark on NVIDIA GeForce RTX 5090, Torch 2.11.0+cu128,
CUDA 12.8. Timings use static output/workspace tensors.

| rows | vocab | op | FlashRT us | PyTorch us | Speedup | Notes |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| 16 | 32,000 | `argmax_bf16` | 8.257 | 12.328 | 1.49x | static output |
| 16 | 32,000 | `accept_partitioned_bf16` | 12.329 | n/a | n/a | static workspace, `parts=8` |
| 16 | 248,320 | `argmax_bf16` | 43.279 | 16.710 | 0.39x | compatibility only; do not use as headline for large vocab |
| 16 | 248,320 | `accept_partitioned_bf16` | 12.342 | n/a | n/a | static workspace, `parts=32` |

For large vocabularies, use `accept_partitioned_bf16`; the single-CTA
`argmax_bf16` path is intended for medium vocabulary rows or compatibility
coverage, not as a 248k-vocab headline.
