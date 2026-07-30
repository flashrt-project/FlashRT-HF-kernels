# Validation: flashrt-spatiotemporal-layout

Required before publishing this package:

1. Source-extension correctness:

   ```bash
   python flashrt-spatiotemporal-layout/tests/test_spatiotemporal_layout.py \
     --backend source \
     --mode full
   ```

2. Source-extension benchmark:

   ```bash
   python flashrt-spatiotemporal-layout/benchmarks/benchmark.py \
     --backend source \
     --shapes all
   ```

   Native parity and compiled-reference gate:

   ```bash
   python flashrt-spatiotemporal-layout/benchmarks/benchmark_native_parity.py \
     --backend source
   ```

3. Kernel-builder artifact build:

   ```bash
   kernel-builder build-and-copy flashrt-spatiotemporal-layout
   ```

4. Built-artifact correctness:

   ```bash
   PYTHONPATH=<artifact-path> \
   python flashrt-spatiotemporal-layout/tests/test_spatiotemporal_layout.py \
     --backend installed \
     --mode full
   ```

5. Built-artifact benchmark:

   ```bash
   python flashrt-spatiotemporal-layout/benchmarks/benchmark.py \
     --backend installed \
     --artifact <artifact-path> \
     --shapes all
   ```

   Repeat `benchmark_native_parity.py --backend installed
   --artifact <artifact-path>`. Acceptance requires exact output, direct-call
   wrapper overhead no greater than 5% or 0.5 us, and CUDA Graph replay
   overhead no greater than 5% or 0.25 us versus the original native entry.

6. Multi-hardware matrix:

   Add hardware claims only after the same correctness and benchmark commands
   pass on that machine.
