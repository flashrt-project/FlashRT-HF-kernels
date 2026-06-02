# Benchmark Results: flashrt-vla-video

These are local RTX 5090 numbers used to select and tune the first showcase
slice. They are suitable for first-batch triage, but not yet a multi-hardware
release benchmark table.

Validated on June 2, 2026.

Environment:

- GPU: NVIDIA GeForce RTX 5090
- PyTorch: 2.9.1+cu128
- CUDA runtime reported by PyTorch: 12.8
- Timing path: FlashRT internal pybind module for source selection, followed by
  a package-local source-extension smoke benchmark.
- Tile sweep: warmup 20, measured iterations 100.
- Tile override: `FLASHRT_QKV_ROPE_BLOCK_SIZE in {128, 256, 512}`.

## Current Triage

- `qkv_split_norm_rope_bf16` is the strongest first showcase candidate. It
  removes packed-QKV split, Q/K RMSNorm, and interleaved RoPE intermediate
  launches, and reaches 24-40x on the short/vision token shapes and 19-29x on
  long video-token shapes in the current RTX 5090 sweep.
- `q_norm_rope_bf16` and `k_norm_rope_v_cache_bf16` are smaller decode
  primitives. They remain useful package APIs and show stable high-20x to
  low-30x speedups versus PyTorch eager for head_dim=128 post-processing.
- NVFP4 fused quantization epilogues are useful but currently measure in the
  2.5-5.2x range versus PyTorch eager plus FlashRT quantization, so they are
  not the first headline if the bar is a 30x-class kernel.

## Tile Policy

The current SM120 default policy is:

- use 512-thread CTAs for `tokens <= 64`;
- use 256-thread CTAs for longer token blocks.

The policy favors short-context and head-count sweep wins while avoiding the
long-token regression seen with 512-thread CTAs at `tokens >= 2520`.

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

## Package-Local QKV Split + Norm + RoPE Sweep

Compiled with the same package-local source extension. Baseline is PyTorch
eager split + RMSNorm + interleaved RoPE. The table below follows the current
default tile policy.

| Shape | Tile | Fused us | PyTorch eager us | Speedup | Max error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| B=1,T=1,H=24,D=128 | 512 | 4.155 | 162.573 | 39.13x | 0.06250 |
| B=1,T=4,H=24,D=128 | 512 | 4.149 | 165.208 | 39.82x | 0.12500 |
| B=1,T=16,H=24,D=128 | 512 | 4.157 | 164.253 | 39.51x | 0.12500 |
| B=1,T=64,H=24,D=128 | 512 | 4.158 | 165.057 | 39.69x | 0.12500 |
| B=1,T=256,H=24,D=128 | 256 | 6.193 | 161.163 | 26.02x | 0.12500 |
| B=1,T=1024,H=24,D=128 | 256 | 12.131 | 235.017 | 19.37x | 0.25000 |
| B=1,T=2520,H=24,D=128 | 256 | 20.546 | 506.212 | 24.64x | 0.12500 |
| B=1,T=4096,H=24,D=128 | 256 | 36.022 | 1043.616 | 28.97x | 0.12500 |

## Head-Count Sweep

These shapes use `tokens=64`, `head_dim=128`, and the current default
512-thread CTA path.

| Shape | Fused us | PyTorch eager us | Speedup | Max error |
| ---: | ---: | ---: | ---: | ---: |
| B=1,T=64,H=8,D=128 | 4.096 | 162.366 | 39.64x | 0.06250 |
| B=1,T=64,H=16,D=128 | 4.134 | 162.510 | 39.31x | 0.12500 |
| B=1,T=64,H=32,D=128 | 4.176 | 165.738 | 39.69x | 0.12500 |
| B=1,T=64,H=48,D=128 | 6.201 | 163.707 | 26.40x | 0.12500 |
