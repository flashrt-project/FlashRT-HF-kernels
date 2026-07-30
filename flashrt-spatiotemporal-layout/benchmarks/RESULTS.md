# Source Benchmark Results

Environment: NVIDIA GeForce RTX 5090 local source-extension build.
Baseline: PyTorch eager tensor layout/reference operations.

| Shape | Tensor | Kernel | FlashRT us | Eager us | vs eager | Verified |
|---|---:|---|---:|---:|---:|---|
| small | `(1, 8, 4, 8, 8)` | ncdhw_to_blc_bf16 | 2.548 | 5.018 | 1.97x | yes |
| small | `(1, 16, 4, 8, 8)` | time_unshuffle2_bf16 | 2.416 | 10.460 | 4.33x | yes |
| small | `(1, 8, 4, 8, 8)` | add_bias_ncdhw_bf16 | 2.400 | 15.491 | 6.45x | yes |
| small | `(1, 8, 4, 8, 8)` | update_cache2_ncdhw_bf16 | 2.461 | 5.337 | 2.17x | yes |
| latent_16 | `(1, 16, 8, 32, 32)` | ncdhw_to_blc_bf16 | 3.910 | 4.960 | 1.27x | yes |
| latent_16 | `(1, 32, 8, 32, 32)` | time_unshuffle2_bf16 | 4.079 | 10.353 | 2.54x | yes |
| latent_16 | `(1, 16, 8, 32, 32)` | add_bias_ncdhw_bf16 | 2.353 | 13.940 | 5.92x | yes |
| latent_16 | `(1, 16, 8, 32, 32)` | update_cache2_ncdhw_bf16 | 2.371 | 5.032 | 2.12x | yes |
| latent_64 | `(1, 64, 4, 32, 32)` | ncdhw_to_blc_bf16 | 4.125 | 4.977 | 1.21x | yes |
| latent_64 | `(1, 128, 4, 32, 32)` | time_unshuffle2_bf16 | 4.109 | 10.314 | 2.51x | yes |
| latent_64 | `(1, 64, 4, 32, 32)` | add_bias_ncdhw_bf16 | 2.403 | 14.793 | 6.16x | yes |
| latent_64 | `(1, 64, 4, 32, 32)` | update_cache2_ncdhw_bf16 | 2.493 | 5.212 | 2.09x | yes |

## Native parity and compile baselines

RTX 5090 source-extension results. Times are microseconds. `Graph` is wrapper
CUDA Graph replay; it is compared with a separately captured raw native
entry. All outputs are bitwise exact. No equivalent single cuDNN/CUTLASS
operation exists for these fused layout/quantization contracts.

| Workload | Native | Wrapper | Wrapper/native | Graph wrapper/native | Eager | compile |
|---|---:|---:|---:|---:|---:|---:|
| latent-small transpose | 2.140 | 2.612 | 1.220 | 2.138 / 2.134 | 4.186 | 22.435 |
| latent-small transpose+bias | 4.141 | 4.138 | 0.999 | 4.092 / 4.090 | 18.186 | 29.065 |
| latent-small transpose+residual | 4.166 | 4.150 | 0.996 | 4.135 / 4.147 | 16.929 | 23.848 |
| latent-small quantize+layout | 4.125 | 4.095 | 0.993 | 4.126 / 4.134 | 18.106 | 22.449 |
| C320 transpose | 8.255 | 8.218 | 0.996 | 8.218 / 8.243 | 47.180 | 26.758 |
| C320 transpose+bias | 10.274 | 10.279 | 1.000 | 10.256 / 10.255 | 79.967 | 34.047 |
| C320 transpose+residual | 14.356 | 14.380 | 1.002 | 14.369 / 14.386 | 137.416 | 27.961 |
| C320 quantize+layout | 10.298 | 10.342 | 1.004 | 10.281 / 10.287 | 88.733 | 26.706 |
| C512 transpose | 10.269 | 10.336 | 1.006 | 10.303 / 10.269 | 75.859 | 28.037 |
| C512 transpose+bias | 12.353 | 12.356 | 1.000 | 12.420 / 12.449 | 125.055 | 37.078 |
| C512 transpose+residual | 20.516 | 20.514 | 1.000 | 20.522 / 20.508 | 221.544 | 29.151 |
| C512 quantize+layout | 16.413 | 16.434 | 1.001 | 16.383 / 16.403 | 119.773 | 28.036 |

The 2.14 us small transpose exposes about 0.47 us of direct Tensor-dispatch
overhead. Its CUDA Graph replay is native-equivalent. It is accepted under the
explicit absolute-overhead rule, not hidden behind an eager-only speedup.
