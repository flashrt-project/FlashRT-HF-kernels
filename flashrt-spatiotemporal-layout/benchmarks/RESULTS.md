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
