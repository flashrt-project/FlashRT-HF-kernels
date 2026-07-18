# Validation: flashrt-fp8-ffn

Required before publishing:

1. Source-extension correctness on RTX 5090:

   ```bash
   python flashrt-fp8-ffn/tests/test_fp8_ffn.py
   ```

2. Source-extension benchmark with eager and compile-stable baselines:

   ```bash
   python flashrt-fp8-ffn/benchmarks/benchmark.py --compile-baseline
   python flashrt-fp8-ffn/benchmarks/benchmark_bf16_entry.py \
     --backend source --shapes all --compile-baseline
   python flashrt-fp8-ffn/benchmarks/benchmark_linear_bias.py \
     --backend source --shapes all --compile-baseline --compare-fvk
   ```

   The BF16-entry matrix covers M `8, 51, 64, 105, 128` for both
   `1152 -> 4304 -> 1152` and `1536 -> 6144 -> 1536`. Production-order input
   quantization and the complete output must be bit-exact against the same
   staged FlashRT ops. The gate also checks explicit `51 -> 64` zero padding.

   Performance qualification is tied to the exact Torch, CUDA, architecture,
   and artifact tuple. Both allocation-free regular calls and explicit CUDA
   Graph replay must be reported. For the RTX 5090 release stack, each M=51
   family must improve over the previous published artifact and the fastest
   production-eligible path must reach at least `1.3x` over both BF16 eager and
   a verified `torch.compile` reference. A result from a different software
   stack cannot substitute for the built-artifact gate.

   The compiled reference must verify against eager output before timing is
   reported. The current compile-stable reference graph-breaks the
   `GELU -> FP8 requant` and final BF16 bias/cast boundaries, while keeping the
   FP8 dequant GEMM regions compiled.

   The linear+bias matrix covers M `1, 8, 51, 64, 105, 128, 256, 512`, QKV
   expansion and output projections. Every row must pass max/mean/p99 absolute
   error, cosine, dtype, invalid-input, `torch.compile(fullgraph=True)`, and
   CUDA Graph replay gates. Performance promotion is separate from functional
   support: M=51 and M=105 must be no slower than the matching production
   FlashRT FP8+bias path and at least `1.3x` faster than BF16 eager. M=1 and M=8
   are diagnostic dispatch rows and must not be advertised as universal wins.

3. Built-artifact validation after the full kernel-builder pass:

   ```bash
   PYTHONPATH=<artifact-path> python flashrt-fp8-ffn/tests/test_fp8_ffn.py --backend installed
   python flashrt-fp8-ffn/benchmarks/benchmark_bf16_entry.py \
     --backend installed --artifact <artifact-path> --shapes all \
     --compile-baseline
   python flashrt-fp8-ffn/benchmarks/benchmark_linear_bias.py \
     --backend installed --artifact <artifact-path> --shapes all \
     --compile-baseline
   ```

   Re-run the full correctness matrix from a cold `get_kernel(..., version=1)`
   download. Verify that the loaded shared-object suffix matches the release
   source commit and test with both the current `kernels` release and the
   Transformers/Diffusers-compatible `kernels==0.12.3` client.

4. Model-block demo:

   ```bash
   python demos/pi05-groot-ffn-epilogue/benchmark.py --backend source --compile-baseline
   ```

   This demo covers epilogue and activation-quant chains where the compiled
   reference is verified separately from the full FP8 FFN package baseline.

5. Multi-hardware matrix:

   Add RTX 5090, RTX 4090/Ada, H100/H200, L4, T4, V100, and workstation cards
   only after those machines run the same correctness and benchmark scripts.
