# Validation: flashrt-fp8-ffn

Required before publishing:

1. Source-extension correctness on RTX 5090:

   ```bash
   python flashrt-fp8-ffn/tests/test_fp8_ffn.py
   ```

2. Source-extension benchmark with eager and compile-stable baselines:

   ```bash
   python flashrt-fp8-ffn/benchmarks/benchmark.py --compile-baseline
   ```

   The compiled reference must verify against eager output before timing is
   reported. The current compile-stable reference graph-breaks the
   `GELU -> FP8 requant` and final BF16 bias/cast boundaries, while keeping the
   FP8 dequant GEMM regions compiled.

3. Built-artifact validation after the full kernel-builder pass:

   ```bash
   PYTHONPATH=<artifact-path> python flashrt-fp8-ffn/tests/test_fp8_ffn.py --backend installed
   ```

4. Model-block demo:

   ```bash
   python demos/pi05-groot-ffn-epilogue/benchmark.py --backend source --compile-baseline
   ```

   This demo covers epilogue and activation-quant chains where the compiled
   reference is verified separately from the full FP8 FFN package baseline.

5. Multi-hardware matrix:

   Add RTX 5090, RTX 4090/Ada, H100/H200, L4, T4, V100, and workstation cards
   only after those machines run the same correctness and benchmark scripts.
