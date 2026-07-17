# benchmarks

Package-level microbenchmarks for FP8 GeGLU/SwiGLU FFN blocks.

The benchmark compares:

- FlashRT Tensor API: `fp8_swiglu_mlp_bf16`
- FlashRT Tensor API: `fp8_geglu_mlp_bf16`
- PyTorch eager reference: FP8 dequant GEMM, `SiLU(gate) * up` or
  `GELU_tanh(gate) * up`, FP8 requant, FP8 dequant GEMM
- Optional `torch.compile` reference after the compiled reference is verified
  against eager

Run:

```bash
python flashrt-fp8-swiglu-ffn/benchmarks/benchmark.py --backend source --shapes headline
python flashrt-fp8-swiglu-ffn/benchmarks/benchmark.py --backend source --shapes all
python flashrt-fp8-swiglu-ffn/benchmarks/benchmark_bf16_entry.py \
  --backend source --activation silu --shapes all --compile-baseline
python flashrt-fp8-swiglu-ffn/benchmarks/benchmark_bf16_entry.py \
  --backend source --activation gelu --shapes all --compile-baseline
```

The BF16-entry benchmark adds BF16 eager, verified full-graph
`torch.compile`, separate Python quantization, kernel-only staging, and
explicit CUDA Graph timing. M=51 rows must reach `>=1.3x` over BF16 eager and
all rows must match the established staged FlashRT path exactly.

The BF16-entry benchmark covers the complete static-quantized SwiGLU or GeGLU
region, including BF16 input quantization, and compares the package call,
explicit CUDA Graph replay, the previous separate quantize-plus-FFN path, FP8
kernel-only time, BF16 PyTorch eager, and a correctness-verified
`torch.compile` baseline:

```bash
python flashrt-fp8-swiglu-ffn/benchmarks/benchmark_bf16_entry.py \
  --backend source --activation silu --shapes all --compile-baseline
python flashrt-fp8-swiglu-ffn/benchmarks/benchmark_bf16_entry.py \
  --backend source --activation gelu --shapes all --compile-baseline
```

Use `--backend installed --artifact <variant-dir>` for a built artifact or
`--backend hub --repo-id flashrt/flashrt-fp8-swiglu-ffn --version 1` for the
Hub package. Raw machine-readable rows can be written with `--output <path>`.
