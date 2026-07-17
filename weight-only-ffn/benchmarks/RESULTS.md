# Results

Installed kernel-builder artifact benchmark on NVIDIA GeForce RTX 5090,
SM120, Torch 2.11.0+cu128. Static weight preparation is excluded from latency.
Each reported timing is the median of three CUDA-event measurement rounds.

| Shape | Precision | Region | Kernel us | Eager us | Compile us | vs eager | vs compile |
|---|---:|---|---:|---:|---:|---:|---:|
| M1 K4096 H11008 N4096 | W4A16 | SwiGLU | 53.3 | 158.2 | 171.4 | 2.97x | 3.22x |
| M1 K4096 H11008 N4096 | W4A16 | GeGLU | 53.3 | 158.1 | 172.1 | 2.97x | 3.23x |
| M1 K4096 H11008 N4096 | W4A16 | GELU | 44.5 | 104.1 | 116.4 | 2.34x | 2.62x |
| M1 K4096 H11008 N4096 | W8A16 | SwiGLU | 91.3 | 158.7 | 173.3 | 1.74x | 1.90x |
| M1 K4096 H11008 N4096 | W8A16 | GeGLU | 91.3 | 158.7 | 173.2 | 1.74x | 1.90x |
| M1 K4096 H11008 N4096 | W8A16 | GELU | 28.2 | 104.0 | 115.5 | 3.69x | 4.10x |
| M2 K4096 H11008 N4096 | W4A16 | SwiGLU | 77.9 | 177.2 | 170.2 | 2.27x | 2.19x |
| M2 K4096 H11008 N4096 | W8A16 | GELU | 34.2 | 119.0 | 115.6 | 3.48x | 3.38x |

The table only includes rows that pass the correctness gate and the package's
production performance threshold. The complete machine-readable sweep is kept
by the release runbook.

The full sweep contains 60 model-shape rows: 39 are accepted by production
auto dispatch and 21 known weak W4/W8 geometries are rejected. Across accepted
rows, auto is at most 1.81% slower than the fastest diagnostic tile. The
minimum accepted speedup in this matrix is 1.22x versus eager and 1.38x versus
the equivalent compiled reference; weak rows are not reported as production
speedups.
