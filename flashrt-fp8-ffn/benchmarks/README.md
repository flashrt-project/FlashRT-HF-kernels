# benchmarks

Package-level microbenchmarks for FP8 GEMM and GELU MLP blocks.

`benchmark_linear_bias.py` measures the reusable FP8 Q/K/V/O-style projection
surface. It covers direct FP8 input, BF16 region entry, explicit CUDA Graph,
BF16 eager, a correctness-verified `torch.compile` reference, and optionally
the original FlashRT FVK FP16/BF16 bias paths.

`benchmark_bf16_entry.py` measures the complete BF16-boundary region against
BF16 eager, verified full-graph `torch.compile`, the former separate Python
quantization path, kernel-only staging, and explicit CUDA Graph replay. M=51
rows are a hard `>=1.3x` BF16-eager gate; all rows require exact staged parity.

The BF16-entry benchmark measures the complete static-quantized region exposed
by `bf16_fp8_gelu_mlp_bf16`. It reports the package call, explicit CUDA Graph
replay, the previous separate quantize-plus-FFN path, FP8 kernel-only time,
BF16 PyTorch eager, and a correctness-verified `torch.compile` baseline.

Run the full mid-M matrix:

```bash
python flashrt-fp8-ffn/benchmarks/benchmark_bf16_entry.py \
  --backend source --shapes all --compile-baseline
python flashrt-fp8-ffn/benchmarks/benchmark_linear_bias.py \
  --backend source --shapes all --compile-baseline --compare-fvk
```

Use `--backend installed --artifact <variant-dir>` for a built artifact or
`--backend hub --repo-id flashrt/flashrt-fp8-ffn --version 1` for the Hub
package. Raw machine-readable rows can be written with `--output <path>`.
