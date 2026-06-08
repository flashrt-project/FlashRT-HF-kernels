# Validation: flashrt-adaptive-norms

Required before publishing this package:

1. Source-extension correctness:

   ```bash
   python flashrt-adaptive-norms/tests/test_adaptive_norms.py \
     --backend source \
     --mode full
   ```

2. Source-extension benchmark:

   ```bash
   python flashrt-adaptive-norms/benchmarks/benchmark.py \
     --backend source \
     --shapes all
   ```

3. Kernel-builder artifact build:

   ```bash
   kernel-builder build-and-copy flashrt-adaptive-norms
   ```

4. Built-artifact correctness:

   ```bash
   PYTHONPATH=<artifact-path> \
   python flashrt-adaptive-norms/tests/test_adaptive_norms.py \
     --backend installed \
     --mode full
   ```

5. Built-artifact benchmark:

   ```bash
   python flashrt-adaptive-norms/benchmarks/benchmark.py \
     --backend installed \
     --artifact <artifact-path> \
     --shapes all
   ```

6. Multi-hardware matrix:

   Add hardware claims only after the same correctness and benchmark commands
   pass on that machine.
