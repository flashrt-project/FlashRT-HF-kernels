# Results

Local source benchmark before unified HF Jobs artifact build.

Environment:

- GPU: NVIDIA GeForce RTX 5090, SM 12.0
- PyTorch: 2.9.1+cu128
- CUDA runtime: 12.8
- Command: `python diffusion-step-ops/benchmarks/benchmark.py --backend source --warmup 100 --iters 1000`
- Last refreshed: 2026-06-20

| Workload | Shape | FlashRT us | PyTorch eager us | Speedup |
|---|---:|---:|---:|---:|
| add_bf16 | `(1024,)` | 3.534 | 12.742 | 3.61x |
| euler_step_bf16 | `(1024,)` | 3.518 | 15.836 | 4.50x |
| cfg_combine_bf16 | `(1024,)` | 2.448 | 24.176 | 9.88x |
| add_bf16 | `(16384,)` | 3.268 | 12.557 | 3.84x |
| euler_step_bf16 | `(16384,)` | 3.610 | 15.604 | 4.32x |
| cfg_combine_bf16 | `(16384,)` | 2.359 | 24.175 | 10.25x |
| add_bf16 | `(2, 16, 32, 64)` | 3.237 | 12.485 | 3.86x |
| euler_step_bf16 | `(2, 16, 32, 64)` | 3.570 | 15.758 | 4.41x |
| cfg_combine_bf16 | `(2, 16, 32, 64)` | 2.316 | 24.020 | 10.37x |
| add_bf16 | `(1, 16, 17, 64, 64)` | 4.096 | 14.343 | 3.50x |
| euler_step_bf16 | `(1, 16, 17, 64, 64)` | 4.100 | 18.437 | 4.50x |
| cfg_combine_bf16 | `(1, 16, 17, 64, 64)` | 4.099 | 30.912 | 7.54x |
| teacher_force_first_frame | `(1, 4, 5, 16, 16)` | 2.304 | 7.870 | 3.42x |
| decode_postprocess | `(1, 4, 5, 16, 16)` | 4.219 | 14.539 | 3.45x |
| teacher_force_first_frame | `(2, 8, 9, 32, 32)` | 2.184 | 3.917 | 1.79x |
| decode_postprocess | `(2, 8, 9, 32, 32)` | 3.959 | 14.637 | 3.70x |
| teacher_force_first_frame | `(1, 16, 17, 64, 64)` | 2.226 | 3.953 | 1.78x |
| decode_postprocess | `(1, 16, 17, 64, 64)` | 6.150 | 16.361 | 2.66x |

Notes:

- `teacher_force_first_frame` is a very small copy-style op and is already
  close to launch-floor. Its main value is CUDA Graph/static-buffer
  compatibility; do not oversell its standalone speedup.
- Installed-artifact benchmark must be regenerated after the unified HF Jobs
  batch build.
