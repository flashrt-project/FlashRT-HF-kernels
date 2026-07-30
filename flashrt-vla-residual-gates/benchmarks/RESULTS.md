# Source Benchmark Results

Environment: NVIDIA GeForce RTX 5090 local source-extension build.
Baseline: PyTorch eager tensor postprocess chain with matching BF16 math.

| Shape | V,A,U rows | Dim | FlashRT us | Eager us | vs eager | p99 abs | Cosine | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| small | 64,8,4 | 1024 | 3.178 | 61.074 | 19.22x | 0.000000 | 0.99999994 | PASS |
| vla_2k | 2520,16,16 | 3072 | 10.326 | 187.622 | 18.17x | 0.000000 | 1.00000000 | PASS |
| vla_4k | 4096,16,16 | 3072 | 20.352 | 361.285 | 17.75x | 0.000000 | 1.00000012 | PASS |

## FP8-gate native and compile comparison

The raw CUDA launcher and Tensor wrapper are bitwise identical. CUDA Graph
rows capture 32 launches and report time per launch.

| Shape | Native us | Wrapper us | Wrapper/native | Graph native us | Graph wrapper us | Graph wrapper/native | Eager us | Compile us | Exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| GROOT `(51,8,4)x1536` | 2.058 | 2.856 | 1.388 | 1.473 | 1.472 | 1.000 | 59.512 | 36.460 | yes |
| VLA `(2520,16,16)x3072` | 10.242 | 10.246 | 1.000 | 8.455 | 8.700 | 1.029 | 215.394 | 49.006 | yes |

The small GROOT row is launch-bound in direct Python calls; graph replay
removes the dispatcher delta. The large VLA row is already at native parity
without capture.
