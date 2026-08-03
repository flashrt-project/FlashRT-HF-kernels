# RTX 5090 source results

Environment: NVIDIA GeForce RTX 5090 (SM120), driver 580.159.03, PyTorch
2.9.1+cu128, CUDA runtime 12.8. Measurements use CUDA events after warmup.

| Workload | M x top-k | N x K | W4A4 region us | W4A4 kernel us | W4A16 us | Per-pair loop us | Region speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| gate_up decode | 1 x 8 | 1024 x 2048 | 10.287 | 6.188 | 6.186 | 65.543 | 6.37x |
| gate_up verify | 7 x 8 | 1024 x 2048 | 30.746 | 26.627 | 22.562 | 458.337 | 14.91x |
| down decode | 8 x 1 | 2048 x 512 | 8.225 | 4.123 | 5.611 | 49.212 | 5.98x |
| down verify | 56 x 1 | 2048 x 512 | 21.932 | 17.971 | 18.485 | 344.385 | 15.70x |

`W4A4 region` includes one batched activation quantization plus one grouped
compute launch. `Per-pair loop` quantizes and launches each routed pair
separately. W4A16 is faster when the only available input is BF16 in three of
four rows; W4A4 kernel-only is competitive or faster when an upstream producer
already emits packed FP4. These results do not claim otherwise.

Full source correctness: 21/21 checks passed. Worst dequantized-contract result
from the random sweep: max abs 0.004084, p99 abs 0.002907, mean abs 0.000507,
cosine 0.9999988. Grouped-vs-native-loop and graph replay checks are bitwise.
