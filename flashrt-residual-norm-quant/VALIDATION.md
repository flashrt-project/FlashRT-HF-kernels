# Validation: flashrt-residual-norm-quant

Required before publishing this package:

1. Source-extension correctness:

   ```bash
   python flashrt-residual-norm-quant/tests/test_residual_norm_quant.py --backend source --mode full
   ```

2. Source-extension benchmark:

   ```bash
   python flashrt-residual-norm-quant/benchmarks/benchmark.py \
     --backend source \
     --shapes all \
     --warmup 3 \
     --iters 10
   ```

3. Kernel-builder artifact build:

   ```bash
   kernel-builder build-and-copy flashrt-residual-norm-quant
   ```

4. Built-artifact correctness:

   ```bash
   PYTHONPATH=<artifact-path> \
   python flashrt-residual-norm-quant/tests/test_residual_norm_quant.py \
     --backend installed \
     --mode full
   ```

5. Built-artifact benchmark:

   ```bash
   python flashrt-residual-norm-quant/benchmarks/benchmark.py \
     --backend installed \
     --artifact <artifact-path> \
     --shapes all
   ```

6. Multi-hardware matrix:

   Add hardware claims only after the same correctness and benchmark commands
   pass on that machine.

## Thor validation

On 2026-08-06, source full passed on NVIDIA Thor (SM110), PyTorch
`2.13.0+cu130`, and CUDA 13.0. The new `residual_add_rms_norm_bf16` entry was
checked both out-of-place and with the production in-place residual contract
across PI0.5, GROOT, and video-prefill shapes. All rows passed the package's
strict BF16 operation-order reference.

Cosmos3-Edge coverage uses `(rows=128, dim=2048)` and additionally requires
bitwise-identical static-output CUDA Graph replay.
