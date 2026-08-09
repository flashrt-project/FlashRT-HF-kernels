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

## Style Broadcast Gate

Both APIs accept `style` as either `(rows, 3*dim)` or `(1, 3*dim)`. The full
source and installed-artifact runs must execute both layouts at rows
`64/2520/4096`. The broadcast path additionally requires bit-identical gate
output, exact residual update, FP8 p99 absolute error zero, CUDA Graph replay,
and installed-wrapper `torch.compile(fullgraph=True)` parity.
