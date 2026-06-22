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
