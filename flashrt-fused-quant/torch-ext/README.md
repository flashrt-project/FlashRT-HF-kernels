# Torch Extension

Planned Tensor wrappers:

- `silu_mul_quant_nvfp4_swizzled_bf16`
- `silu_mul_merged_quant_nvfp4_swizzled_bf16`
- `residual_rmsnorm_quant_nvfp4_swizzled_bf16`
- `rmsnorm_quant_nvfp4_sfa_fp16`
- `residual_rmsnorm_quant_nvfp4_sfa_fp16`

Do not expose model-specific names even when the source provenance is
model-specific.
