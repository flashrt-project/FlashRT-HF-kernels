# RTX 5090 source gate

CUDA 13.0, PyTorch 2.11 development environment, contiguous NHD, causal
Hq/Hkv=32/8, D=128. Baselines consume BF16 conversions of the exact FP8 input;
conversion is prepared outside timing. Output allocation is also outside the
FlashRT timing. Values are microseconds.

| S | FlashRT | BF16 SDPA eager | BF16 SDPA compile | vs eager | vs compile |
|---:|---:|---:|---:|---:|---:|
| 256 | 14.36 | 18.47 | 30.30 | 1.29x | 2.11x |
| 512 | 24.59 | 30.74 | 30.78 | 1.25x | 1.25x |
| 1024 | 63.69 | 84.07 | 84.46 | 1.32x | 1.33x |
| 2048 | 178.34 | 237.84 | 238.25 | 1.33x | 1.34x |
| 4096 | 570.63 | 790.22 | 790.05 | 1.38x | 1.38x |

S=128 measured 12.32 us versus 10.28 us eager and is therefore intentionally
outside the public v1 support envelope. Built-artifact numbers will replace the
source-gate table after HF Jobs publication.
