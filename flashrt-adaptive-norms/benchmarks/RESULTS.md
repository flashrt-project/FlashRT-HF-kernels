# Source Benchmark Results

Environment: NVIDIA GeForce RTX 5090 local source-extension build.
Baseline: PyTorch eager tensor reference with matching BF16/FP8 math contract.

| Shape | Rows,Dim | Kernel | FlashRT us | Eager us | vs eager | p99 abs | Cosine | Status |
|---|---:|---|---:|---:|---:|---:|---:|---|
| small | 64,1024 | ada_rms_norm_style_bf16 | 4.172 | 61.407 | 14.72x | 0.000000 | 0.99999988 | PASS |
| small | 64,1024 | gate_residual_ada_norm_fp8_static_bf16 | 4.150 | 81.997 | 19.76x | 0.000000 | 1.00000000 | PASS |
| vla_2k | 2520,3072 | ada_rms_norm_style_bf16 | 15.404 | 265.453 | 17.23x | 0.000000 | 0.99999994 | PASS |
| vla_2k | 2520,3072 | gate_residual_ada_norm_fp8_static_bf16 | 58.822 | 431.002 | 7.33x | 0.000000 | 1.00000000 | PASS |
| vla_4k | 4096,3072 | ada_rms_norm_style_bf16 | 79.392 | 594.854 | 7.49x | 0.000000 | 1.00000000 | PASS |
| vla_4k | 4096,3072 | gate_residual_ada_norm_fp8_static_bf16 | 119.475 | 901.242 | 7.54x | 0.000000 | 1.00000000 | PASS |

## Thor Style-Broadcast Update

Source benchmark on NVIDIA Thor, PyTorch 2.13.0+cu130, August 8, 2026. The
single-row style path removes the expanded style tensor while preserving the
same public operation.

| Shape | Kernel | Per-row us | Broadcast us | Broadcast vs eager | P99 abs |
|---|---|---:|---:|---:|---:|
| `2520x3072` | AdaRMS style | 407.806 | 202.440 | 15.11x | 0 |
| `2520x3072` | gate+residual+AdaRMS to FP8 | 592.928 | 398.299 | 12.58x | 0 |
| `4096x3072` | AdaRMS style | 604.960 | 306.514 | 14.79x | 0 |
| `4096x3072` | gate+residual+AdaRMS to FP8 | 864.846 | 553.539 | 13.69x | 0 |

The full source gate passed both style layouts. Residual and gate tensors were
exact; FP8 p99 absolute error was zero across all tested rows.
