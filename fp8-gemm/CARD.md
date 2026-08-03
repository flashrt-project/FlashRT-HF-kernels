# flashrt/fp8-gemm

FlashRT native CUDA FP8 GEMV/GEMM kernels for low-latency transformer and
diffuser linear layers.

The block-128 scaled API supports Ada `sm_89` and Blackwell `sm_120a`.
The per-tensor API supports Blackwell `sm_110a` (Jetson AGX Thor) and
`sm_120a`. SM110 uses the production FlashRT Sq/T1/Wide CUTLASS family and has
been swept across PI0.5, GROOT, Cosmos Edge, and LingBot VLA projection shapes.

The block-128 scaled API also exposes a portable pure-SIMT reference
(`portable_fp8_blockwise_simt.cu`) for `sm_110a`, which has no native
blockwise backend. SM89 and SM120 keep their native block-scaled kernels;
the SIMT reference is selected only on non-SM89/SM120 devices. Set
`FLASHRT_FORCE_SIMT=1` to route any device through the SIMT reference
(used by the correctness test to validate parity against the native path).

## Functions

- `fp8_linear_bf16(input, weight, alpha=1.0, out=None, variant=0)`
- `fp8_linear_residual_bf16(input, weight, residual, alpha=1.0, variant=0)`
- `fp8_blockwise_linear_bf16(input, weight, input_scale, weight_scale, out=None)`
- `select_fp8_linear_tile(m, n, k, variant=0)`

On SM110, keep `variant=0` for the tuned public dispatcher. Variants `1`, `2`,
and `3` force Sq, T1, and Wide respectively for diagnostics.

See the repository README for shape contracts, validation status, and examples.
