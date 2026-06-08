# benchmarks

Package-level microbenchmarks for FP8 SwiGLU FFN blocks.

The benchmark compares:

- FlashRT Tensor API: `fp8_swiglu_mlp_bf16`
- PyTorch eager reference: FP8 dequant GEMM, `SiLU(gate) * up`, FP8 requant,
  FP8 dequant GEMM
- Optional `torch.compile` reference after the compiled reference is verified
  against eager

Run:

```bash
python flashrt-fp8-swiglu-ffn/benchmarks/benchmark.py --backend source --shapes headline
python flashrt-fp8-swiglu-ffn/benchmarks/benchmark.py --backend source --shapes all
```
