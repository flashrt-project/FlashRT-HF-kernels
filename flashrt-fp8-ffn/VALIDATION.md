# Validation: flashrt-fp8-ffn

Required before publishing:

1. Source-extension correctness on RTX 5090:

   ```bash
   python flashrt-fp8-ffn/tests/test_fp8_ffn.py
   ```

2. Source-extension benchmark with eager and `torch.compile` baselines:

   ```bash
   python flashrt-fp8-ffn/benchmarks/benchmark.py --compile-baseline
   ```

3. Built-artifact validation after the full kernel-builder pass:

   ```bash
   PYTHONPATH=<artifact-path> python flashrt-fp8-ffn/tests/test_fp8_ffn.py --backend installed
   ```

4. Model-block demo:

   ```bash
   python demos/pi05-groot-ffn-epilogue/benchmark.py --backend source --compile-baseline
   ```

5. Multi-hardware matrix:

   Add RTX 5090, RTX 4090/Ada, H100/H200, L4, T4, V100, and workstation cards
   only after those machines run the same correctness and benchmark scripts.
