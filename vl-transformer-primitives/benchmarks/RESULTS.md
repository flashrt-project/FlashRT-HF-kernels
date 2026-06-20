# Results

Local source benchmark before HF Jobs artifact build.

Environment:

- GPU: NVIDIA GeForce RTX 5090, SM 12.0
- PyTorch: 2.9.1+cu128
- CUDA runtime: 12.8
- Command: `python vl-transformer-primitives/benchmarks/benchmark.py --backend source --warmup 30 --iters 300`

| Workload | Shape | FlashRT us | PyTorch eager us | Speedup |
|---|---:|---:|---:|---:|
| q_norm_rope | heads=1, d=128 | 4.039 | 61.247 | 15.16x |
| k_norm_rope_vwrite | heads=1, d=128 | 4.323 | 62.407 | 14.44x |
| q_norm_rope | heads=4, d=128 | 3.383 | 61.848 | 18.28x |
| k_norm_rope_vwrite | heads=4, d=128 | 4.524 | 65.275 | 14.43x |
| q_norm_rope | heads=8, d=128 | 3.805 | 62.761 | 16.49x |
| k_norm_rope_vwrite | heads=8, d=128 | 4.520 | 65.507 | 14.49x |
| q_norm_rope | heads=16, d=128 | 3.792 | 62.492 | 16.48x |
| k_norm_rope_vwrite | heads=16, d=128 | 4.671 | 66.917 | 14.33x |
| q_norm_rope | heads=32, d=128 | 3.876 | 62.451 | 16.11x |
| k_norm_rope_vwrite | heads=32, d=128 | 4.742 | 65.769 | 13.87x |
| q_norm_rope | heads=40, d=128 | 3.832 | 61.812 | 16.13x |
| k_norm_rope_vwrite | heads=40, d=128 | 4.502 | 65.916 | 14.64x |
| avg_pool_vision | nv=1, h=16, w=16, dim=1024, pool=2 | 4.464 | 12.997 | 2.91x |
| avg_pool_vision | nv=2, h=16, w=16, dim=1152, pool=2 | 4.328 | 13.638 | 3.15x |
| avg_pool_vision | nv=4, h=16, w=16, dim=2048, pool=4 | 4.318 | 12.957 | 3.00x |
| avg_pool_vision | nv=2, h=32, w=32, dim=1024, pool=4 | 4.542 | 13.383 | 2.95x |

Installed-artifact benchmark must be regenerated after HF Jobs publishes the
package artifact.
