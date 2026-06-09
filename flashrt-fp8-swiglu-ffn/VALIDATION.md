# Validation: flashrt-fp8-swiglu-ffn

Required before publishing this package:

1. Source-extension correctness:

   ```bash
   python flashrt-fp8-swiglu-ffn/tests/test_fp8_swiglu_ffn.py --backend source --mode full
   ```

   The test covers:

   - `fp8_gemm_bf16`
   - `silu_mul_merged_quantize_fp8_static_bf16`
   - `gelu_mul_merged_quantize_fp8_static_bf16`
   - `fp8_swiglu_mlp_bf16`
   - `fp8_geglu_mlp_bf16`
   - invalid shape rejection

   The full MLP ops must be exact against the staged kernel composition
   (`max_abs=0`). Activation-quantization checks against PyTorch eager are
   distributional because CUDA libdevice and PyTorch eager can land on opposite
   sides of FP8 bin boundaries for a small fraction of transcendental
   activation values.

2. Source-extension benchmark:

   ```bash
   python flashrt-fp8-swiglu-ffn/benchmarks/benchmark.py \
     --backend source \
     --shapes all \
     --warmup 5 \
     --iters 20
   ```

3. Kernel-builder artifact build:

   ```bash
   kernel-builder build-and-copy flashrt-fp8-swiglu-ffn
   ```

4. Built-artifact correctness:

   ```bash
   PYTHONPATH=<artifact-path> \
   python flashrt-fp8-swiglu-ffn/tests/test_fp8_swiglu_ffn.py \
     --backend installed \
     --mode full
   ```

5. Built-artifact benchmark:

   ```bash
   python flashrt-fp8-swiglu-ffn/benchmarks/benchmark.py \
     --backend installed \
     --artifact <artifact-path> \
     --shapes all \
     --warmup 5 \
     --iters 20
   ```

6. Model-stage demo:

   Run the PI0.5/GROOT HF-runtime demo using preloaded kernels, resident
   weights/scales, explicit scratch buffers, and CUDA Graph capture for static
   shape loops. Do not compare against FlashRT's fully optimized serving
   runtime as the public baseline; compare against official PyTorch model
   paths or clearly labeled model-stage references.

7. Multi-hardware matrix:

   Add hardware claims only after the same correctness and benchmark commands
   pass on that machine.
