# Validation: flashrt-vla-residual-gates

Required before publishing this package:

1. Source-extension correctness:

   ```bash
   python flashrt-vla-residual-gates/tests/test_vla_residual_gates.py \
     --backend source \
     --mode full
   ```

2. Source-extension benchmark:

   ```bash
   python flashrt-vla-residual-gates/benchmarks/benchmark.py \
     --backend source \
     --shapes all
   ```

3. Kernel-builder artifact build:

   ```bash
   kernel-builder build-and-copy flashrt-vla-residual-gates
   ```

4. Built-artifact correctness:

   ```bash
   PYTHONPATH=<artifact-path> \
   python flashrt-vla-residual-gates/tests/test_vla_residual_gates.py \
     --backend installed \
     --mode full
   ```

5. Built-artifact benchmark:

   ```bash
   python flashrt-vla-residual-gates/benchmarks/benchmark.py \
     --backend installed \
     --artifact <artifact-path> \
     --shapes all
   ```

6. Multi-hardware matrix:

   Add hardware claims only after the same correctness and benchmark commands
   pass on that machine.
