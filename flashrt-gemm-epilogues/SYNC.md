# Source Sync

Upstream source: `../official/FlashRT`

Initial upstream commit: `220746a`

Synced files:

- `csrc/gemm/gemm_runner.cu` (`GemmRunner::bf16_nn_bias_gelu` logic only)
- `csrc/quantize/bias_gelu_quantize_fp8.cu`
- `csrc/quantize/bias_gelu_quantize_fp8.cuh`
- `csrc/quantize/awq_quant_fp8_static_bf16.cu`
- `csrc/quantize/awq_quant_fp8_static_bf16.cuh`

Local package files:

- `csrc/bf16_gemm_bias_gelu.cu`
- `csrc/bf16_gemm_bias_gelu.cuh`
- `csrc/bias_gelu_quantize_fp8.cu`
- `csrc/bias_gelu_quantize_fp8.cuh`
- `csrc/channel_scale_quantize_fp8.cu`
- `csrc/channel_scale_quantize_fp8.cuh`
- `torch-ext/torch_binding.cpp`
- `torch-ext/torch_binding.h`

Local edits:

- Kept the CUDA launcher package-local and independent of FlashRT runtime.
- Renamed the AWQ-specific launcher to a generic channel-scale quantization
  public API.
- Added Tensor-based Torch op bindings.
- Added no-bias public wrapper around the same launcher.
- Added the first full GEMM wrapper:
  `bf16_gemm_bias_gelu(a, b, bias, out=None)`.
- Added the adjacent no-activation wrapper:
  `bf16_gemm_bias(a, b, bias, out=None)`.
- Replaced `GemmRunner` ownership with a package-local per-device cuBLASLt
  runtime and 32 MiB workspace.
- Uses the same column-major view trick as FlashRT's FP8 cuBLASLt path:
  public row-major `D(M,N)` is treated as cuBLASLt `D^T(N,M)`, so the call
  computes `B^T @ A^T` internally while preserving the public `(M,K) @ (K,N)`
  tensor contract.
- Removed pointer-only public API exposure.

Candidate source areas:

- `csrc/gemm/`
- `csrc/gemm/fp4/`
- `csrc/quantize/bias_gelu_quantize_fp8.*`
- `csrc/quantize/awq_quant_fp8_static_bf16.*`
- selected declarations from `csrc/bindings.cpp` only as API references

## Required Refactor

- Replace pybind pointer wrappers with Tensor-based Torch op bindings.
- Move shape, dtype, and device validation into `torch-ext/torch_binding.cpp`.
- Keep CUTLASS dependencies declared in `build.toml`.
- Avoid referencing FlashRT `third_party/cutlass`.

## First Source Slice

Implemented first source slice:

```text
bf16_gemm_bias_gelu(a, b, bias, out=None) -> Tensor
bf16_gemm_bias(a, b, bias, out=None) -> Tensor
bias_gelu_quantize_fp8_static_bf16(input, bias, scale, out=None) -> Tensor
gelu_quantize_fp8_static_bf16(input, scale, out=None) -> Tensor
channel_scale_quantize_fp8_static_bf16(input, channel_scale, scale, out=None) -> Tensor
```

Next expand to quantized GEMM plus epilogue wrappers.

Attempted but not promoted:

- `fp8_gemm_bias_gelu_*` cuBLASLt wrappers compile locally but return
  `CUBLAS_STATUS_NOT_SUPPORTED` from `cublasLtMatmulAlgoGetHeuristic` on the
  local RTX 5090 / PyTorch 2.9.1+cu128 environment, even after matching the
  column-major layout used by FlashRT's FP8 helper. Keep FP8 GEMM promotion
  pending until we select the CUTLASS/DeepGEMM-style scaled FP8 path.
