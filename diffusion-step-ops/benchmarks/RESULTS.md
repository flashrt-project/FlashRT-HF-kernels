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

## Native, Tensor wrapper, eager and compile

The Cosmos Edge-derived additions were also compared directly with their
original FlashRT CUDA launchers. Direct calls include their respective host
dispatch paths. CUDA Graph rows capture 32 launches and report time per launch,
which isolates the actual hot-path device work.

| Workload | Native us | Wrapper us | Wrapper/native | Graph native us | Graph wrapper us | Graph wrapper/native | Eager us | Compile us | Native exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| UniPC `B1x16x17x8x8` | 2.192 | 3.700 | 1.688 | 1.089 | 1.089 | 1.000 | 60.049 | 25.655 | yes |
| pack tail `1024 -> 4096` | 2.055 | 2.301 | 1.119 | 0.896 | 0.960 | 1.072 | 5.939 | 21.018 | yes |
| extract tail `4096 -> 1024` | 1.999 | 2.379 | 1.190 | 0.896 | 0.960 | 1.072 | 4.425 | 20.910 | yes |
| bias + zero tail `105x257` | 4.175 | 2.530 | 0.606 | 2.177 | 1.217 | 0.559 | 16.870 | 21.922 | yes |
| two-stage bias `105x257` | 4.248 | 2.247 | 0.529 | 2.175 | 1.216 | 0.559 | 26.296 | 22.620 | yes |

The direct-call ratios for these launch-floor operations are Python/Tensor
dispatcher overhead, not slower CUDA kernels. The CUDA Graph numbers are the
relevant runtime-pipeline gate.

The two bias rows intentionally improve on the original native contract:
FlashRT first copied the input and then launched an in-place bias kernel; the
generic Hub API reads the input and writes the final output in one launch.
