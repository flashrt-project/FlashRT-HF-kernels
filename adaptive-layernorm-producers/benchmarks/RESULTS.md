# adaptive-layernorm-producers Benchmark Results

Local source-build benchmark:

- GPU: NVIDIA GeForce RTX 5090
- Driver: 580.82.07
- Runtime: local Torch 2.11 / CUDA 12.8 environment
- Command: `python adaptive-layernorm-producers/benchmarks/benchmark.py --backend source --iters 100`
- Baseline: PyTorch eager producer chain with equivalent operations.
- Status: source correctness passed before benchmark. Refresh this table after
  installed-artifact validation on each target hardware.

| Shape | Rows | Dim | AdaLN->FP8 us | Eager chain us | Speedup | LN->FP8 us | Eager LN chain us | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decode_action | 16 | 2048 | 4.117 | 65.010 | 15.79x | 3.962 | 48.006 | 12.12x |
| wan_video_short | 64 | 3072 | 4.133 | 63.425 | 15.35x | 4.117 | 47.826 | 11.62x |
| wan_video_ctx | 256 | 3072 | 4.140 | 69.821 | 16.86x | 4.108 | 55.611 | 13.54x |
| wan_video_2k | 2520 | 3072 | 12.330 | 263.276 | 21.35x | 10.267 | 218.926 | 21.32x |
| wan_video_4k | 4096 | 3072 | 18.465 | 463.733 | 25.11x | 16.412 | 394.908 | 24.06x |

## Six-way modulation producer

The new `adaln_modulation6_bf16` path was measured against its raw CUDA
launcher, the Tensor wrapper, eager, and `torch.compile`. Graph figures capture
32 identical launches per graph and divide replay time by 32, avoiding a graph
launch-floor artifact for the smallest row.

| Shape | Native us | Wrapper us | Graph native us | Graph wrapper us | Eager us | Compile us | Wrapper/native | Graph wrapper/native |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GROOT DiT B1 S51 D1536 | 2.221 | 3.052 | 1.536 | 1.663 | 48.804 | 43.536 | 1.374 | 1.083 |
| Motus B1 S2520 D3072 | 175.952 | 176.183 | 165.263 | 165.513 | 209.040 | 176.665 | 1.001 | 1.002 |
| video long B1 S5070 D3072 | 347.251 | 354.773 | 344.858 | 344.592 | 582.678 | 354.059 | 1.022 | 0.999 |

The direct S51 row exposes about `0.83 us` of dispatcher overhead. The
multi-launch CUDA Graph path reduces the absolute gap to `0.127 us`. Both
figures are retained; the direct row is not described as native-equivalent.
