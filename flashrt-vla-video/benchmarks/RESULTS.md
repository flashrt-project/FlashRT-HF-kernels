# Benchmark Results: flashrt-vla-video

These are preliminary local numbers used to select the first showcase slice.
They are not yet a stable release benchmark table.

Validated on June 2, 2026.

Environment:

- GPU: NVIDIA GeForce RTX 5090
- PyTorch: 2.9.1+cu128
- CUDA runtime reported by PyTorch: 12.8
- Timing path: FlashRT internal pybind module for source selection, followed by
  a package-local source-extension smoke benchmark.

## Current Triage

- `q_norm_rope_bf16` and `k_norm_rope_v_cache_bf16` are the first showcase
  candidates. They show stable 29-34x speedups versus PyTorch eager for
  head_dim=128 decode post-processing.
- `qkv_split_norm_rope_bf16` is the next package-local showcase candidate. It
  shows 21-38x speedups versus PyTorch eager for packed video/VLA QKV
  post-processing.
- NVFP4 fused quantization epilogues are useful but currently measure in the
  2.5-5.2x range versus PyTorch eager plus FlashRT quantization, so they are
  not the first headline if the bar is a 30x-class kernel.

## Q RMSNorm + RoPE + Stage Write

| n_heads | Fused us | PyTorch eager us | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 2.233 | 76.245 | 34.14x |
| 4 | 2.284 | 66.647 | 29.18x |
| 8 | 2.062 | 66.312 | 32.16x |
| 16 | 2.078 | 66.409 | 31.96x |
| 32 | 2.212 | 66.513 | 30.08x |
| 48 | 2.194 | 66.515 | 30.32x |

## K RMSNorm + RoPE + K-Cache Write + V-Cache Copy

| n_heads | Fused us | PyTorch eager us | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 2.062 | 68.507 | 33.22x |
| 4 | 2.075 | 69.296 | 33.39x |
| 8 | 2.154 | 69.255 | 32.15x |
| 16 | 2.313 | 69.502 | 30.05x |
| 32 | 2.062 | 70.425 | 34.15x |
| 48 | 2.075 | 70.232 | 33.85x |

## Package-Local Source Smoke

The package-local source extension was compiled with
`torch.utils.cpp_extension.load` using:

- `torch-ext/torch_binding.cpp`
- `csrc/q_norm_rope_bf16.cu`

| Shape | Q fused us | Q eager us | Q speedup | K fused us | K eager us | K speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| heads=1 | 2.639 | 71.105 | 26.95x | 2.564 | 73.607 | 28.71x |
| heads=8 | 2.464 | 71.944 | 29.20x | 2.555 | 74.246 | 29.06x |
| heads=48 | 2.454 | 75.614 | 30.81x | 2.667 | 77.466 | 29.05x |

## Package-Local QKV Split + Norm + RoPE Smoke

Compiled with the same package-local source extension. Baseline is PyTorch
eager split + RMSNorm + interleaved RoPE.

| Tokens | Fused us | PyTorch eager us | Speedup |
| ---: | ---: | ---: | ---: |
| 4 | 4.473 | 168.442 | 37.66x |
| 64 | 4.856 | 162.639 | 33.49x |
| 256 | 6.209 | 158.634 | 25.55x |
| 1024 | 10.812 | 229.836 | 21.26x |
| 2520 | 20.552 | 504.120 | 24.53x |
