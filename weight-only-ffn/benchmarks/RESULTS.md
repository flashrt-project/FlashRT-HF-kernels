# Results

Static weight preparation is excluded from latency. Each timing is the median
of three CUDA-event measurement rounds after warmup. PyTorch eager and warmed
`torch.compile(mode="max-autotune-no-cudagraphs")` are both measured; the
stronger one is the production baseline.

## RTX 5090 SM120

Source release candidate, Torch 2.9.1+cu128, 40 warmup iterations and 200
measured iterations per round.

| Shape | Precision | Region | Kernel us | Eager us | Compile us | vs eager | vs compile |
|---|---:|---|---:|---:|---:|---:|---:|
| M1 K4096 H11008 N4096 | W4A16 | SwiGLU | 53.1 | 158.4 | 164.0 | 2.98x | 3.09x |
| M1 K4096 H11008 N4096 | W4A16 | GELU | 44.0 | 104.1 | 109.0 | 2.37x | 2.48x |
| M1 K4096 H11008 N4096 | W8A16 | SwiGLU | 90.2 | 158.5 | 165.8 | 1.76x | 1.84x |
| M1 K4096 H11008 N4096 | W8A16 | GELU | 28.1 | 104.4 | 110.5 | 3.71x | 3.93x |
| M2 K4096 H11008 N4096 | W8A16 | GELU | 31.6 | 119.0 | 116.5 | 3.77x | 3.69x |
| M1 K4096 N11008 | W4A16 | Linear | 18.4 | 38.9 | 55.4 | 2.11x | 3.01x |
| M1 K4096 N11008 | W8A16 | Linear | 11.8 | 43.0 | 39.0 | 3.66x | 3.32x |

The full sweep contains 76 rows: 51 are accepted by production auto dispatch
and 25 known weak W4/W8 geometries are rejected. Every accepted row beats the
stronger eager/compile baseline by at least 2%. The minimum accepted speedup
is 1.22x versus eager and 1.37x versus compile. Auto is at most 2.82% slower
than the fastest explicit diagnostic tile.

## NVIDIA Thor SM110

CUDA 13 builder-generated installed artifact, Torch 2.11.0+cu130, 40 warmup
iterations and 200 measured iterations per round.

| Shape | Precision | Region | Kernel us | Eager us | Compile us | vs eager | vs compile |
|---|---:|---|---:|---:|---:|---:|---:|
| M1 K4096 H11008 N4096 | W4A16 | SwiGLU | 528.1 | 1088.4 | 1159.7 | 2.06x | 2.20x |
| M1 K4096 H11008 N4096 | W4A16 | GELU | 365.3 | 733.1 | 733.0 | 2.01x | 2.01x |
| M1 K4096 H11008 N4096 | W8A16 | SwiGLU | 534.2 | 1084.6 | 1086.2 | 2.03x | 2.03x |
| M1 K4096 H11008 N4096 | W8A16 | GELU | 363.7 | 731.4 | 734.2 | 2.01x | 2.02x |
| M2 K4096 H11008 N4096 | W8A16 | GELU | 363.5 | 733.3 | 735.3 | 2.02x | 2.02x |
| M1 K4096 N11008 | W4A16 | Linear | 161.8 | 357.8 | 341.9 | 2.21x | 2.11x |
| M1 K4096 N11008 | W8A16 | Linear | 174.8 | 356.1 | 341.5 | 2.04x | 1.95x |

The full sweep contains 76 rows: 33 are accepted and 43 weak geometries are
rejected. Every accepted row beats the stronger baseline by at least 2%. The
minimum accepted speedup is 1.19x versus both eager and compile. Auto is at
most 0.76% slower than the fastest explicit diagnostic tile.

Only accepted rows are reported as production speedups. Rejected rows remain
in the machine-readable release sweep and raise from `variant=0`; explicit
variants remain available for diagnostics and correctness isolation.
