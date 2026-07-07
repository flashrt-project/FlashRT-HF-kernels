# Upstream Sync

Source family:

- `official/FlashRT/csrc/kernels/quantize.cu`
- `official/FlashRT/csrc/kernels/norm.cu`
- `official/FlashRT/csrc/gemm/cutlass_sm80_int8_rowwise.cu`
- `official/FlashRT/csrc/gemm/cutlass_sm80_int8_rowwise_t64x128.cu`
- `official/FlashRT/csrc/gemm/cutlass_sm80_int8_silu_gated.cu`

This package keeps a model-neutral Tensor API and does not mirror FlashRT
runtime pointer bindings.
