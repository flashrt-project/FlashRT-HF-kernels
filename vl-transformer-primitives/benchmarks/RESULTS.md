# Results

Local source benchmark before HF Jobs artifact build.

Environment:

- GPU: NVIDIA GeForce RTX 5090, SM 12.0
- PyTorch: 2.9.1+cu128
- CUDA runtime: 12.8
- Command: `python vl-transformer-primitives/benchmarks/benchmark.py --backend source --warmup 100 --iters 1000`
- Last refreshed: 2026-06-20

| Workload | Shape | FlashRT us | PyTorch eager us | Speedup |
|---|---:|---:|---:|---:|
| q_norm_rope | heads=1, d=128 | 3.874 | 60.964 | 15.74x |
| k_norm_rope_vwrite | heads=1, d=128 | 4.583 | 64.066 | 13.98x |
| q_norm_rope | heads=4, d=128 | 3.756 | 61.136 | 16.28x |
| k_norm_rope_vwrite | heads=4, d=128 | 4.486 | 64.894 | 14.47x |
| q_norm_rope | heads=8, d=128 | 3.772 | 60.606 | 16.07x |
| k_norm_rope_vwrite | heads=8, d=128 | 4.533 | 64.940 | 14.33x |
| q_norm_rope | heads=16, d=128 | 3.794 | 61.910 | 16.32x |
| k_norm_rope_vwrite | heads=16, d=128 | 4.527 | 66.525 | 14.70x |
| q_norm_rope | heads=32, d=128 | 3.759 | 60.816 | 16.18x |
| k_norm_rope_vwrite | heads=32, d=128 | 4.531 | 64.177 | 14.16x |
| q_norm_rope | heads=40, d=128 | 3.765 | 60.923 | 16.18x |
| k_norm_rope_vwrite | heads=40, d=128 | 4.636 | 64.385 | 13.89x |
| avg_pool_vision | nv=1, h=16, w=16, dim=1024, pool=2 | 4.276 | 12.927 | 3.02x |
| avg_pool_vision | nv=2, h=16, w=16, dim=1152, pool=2 | 4.319 | 13.174 | 3.05x |
| avg_pool_vision | nv=4, h=16, w=16, dim=2048, pool=4 | 4.154 | 12.636 | 3.04x |
| avg_pool_vision | nv=2, h=32, w=32, dim=1024, pool=4 | 4.232 | 12.799 | 3.02x |

Installed-artifact benchmark must be regenerated after HF Jobs publishes the
package artifact.
