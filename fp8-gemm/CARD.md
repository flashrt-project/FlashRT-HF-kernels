# flashrt/fp8-gemm

FlashRT native CUDA FP8 GEMV/GEMM kernels for low-latency transformer and
diffuser linear layers.

The block-128 scaled API supports Ada `sm_89` and Blackwell `sm_120a`.
Per-tensor decode and small-M APIs target Blackwell `sm_120a`.

## Functions

- `fp8_linear_bf16(input, weight, alpha=1.0, out=None, variant=0)`
- `fp8_linear_residual_bf16(input, weight, residual, alpha=1.0, variant=0)`
- `fp8_blockwise_linear_bf16(input, weight, input_scale, weight_scale, out=None)`
- `select_fp8_linear_tile(m, n, k, variant=0)`

See the repository README for shape contracts, validation status, and examples.
