# RTX 5090 built-artifact results

Environment: NVIDIA GeForce RTX 5090 (SM120), driver 580.159.03, PyTorch
2.11.0+cu128, CUDA runtime 12.8. Artifact variant:
`torch211-cxx11-cu128-x86_64-linux`, source commit `4baeadf`. Measurements use
CUDA events after warmup.

| Workload | M x top-k | N x K | W4A4 region us | W4A4 kernel us | W4A16 us | Per-pair loop us | Region speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| gate_up decode | 1 x 8 | 1024 x 2048 | 12.286 | 8.177 | 6.204 | 65.612 | 5.34x |
| gate_up verify | 7 x 8 | 1024 x 2048 | 32.785 | 28.691 | 24.605 | 459.022 | 14.00x |
| down decode | 8 x 1 | 2048 x 512 | 10.259 | 6.162 | 6.169 | 64.847 | 6.32x |
| down verify | 56 x 1 | 2048 x 512 | 22.540 | 18.449 | 24.602 | 451.777 | 20.04x |

`W4A4 region` includes one batched activation quantization plus one grouped
compute launch. `Per-pair loop` quantizes and launches each routed pair
separately. On this cu128 artifact W4A16 wins gate-up, the two kernel-only paths
tie for down decode, and W4A4 wins down verify. The complete W4A4 region still
removes the legacy launch storm, but lower precision is not presented as a
universal per-kernel winner.

Full source correctness: 21/21 checks passed. Worst dequantized-contract result
from the random sweep: max abs 0.004084, p99 abs 0.002907, mean abs 0.000507,
cosine 0.9999988. Grouped-vs-native-loop and graph replay checks are bitwise.
