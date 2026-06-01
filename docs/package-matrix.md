# Package Matrix

| Package | First public APIs | Upstream FlashRT areas | Main baseline | Hub maturity target |
| --- | --- | --- | --- | --- |
| `flashrt-gemm-epilogues` | `bias_gelu_quantize_fp8_static_bf16`, `channel_scale_quantize_fp8_static_bf16`, then `fp8_linear_bias_gelu` | `csrc/gemm`, `csrc/gemm/fp4`, selected quant epilogues | PyTorch Linear plus elementwise ops | First package to build |
| `flashrt-fused-quant` | `rmsnorm_quant`, `residual_rmsnorm_quant`, `swiglu_quant`, `qkv_rope_split` | `csrc/kernels`, `csrc/quantize`, `csrc/fused_fp4` | PyTorch eager reference | First/second package |
| `flashrt-nvfp4` | `quantize_nvfp4`, `dequantize_nvfp4`, `reshape_sfa`, `sfa_size_bytes` | `csrc/quantize`, `csrc/gemm/fp4` | PyTorch dequant reference and FlashRT reference | Second package |
| `flashrt-smallm-gemm` | `smallm_fp8_gemm`, `smallm_nvfp4_gemm`, `splitk_decode_gemv` | `csrc/gemm/fp8_smallM*`, small-M matvec/matmul files | cuBLASLt or generic CUTLASS path | Second/third package |
| `flashrt-vla-video` | `patch_embed_fused`, `video_conv_lowbit`, `dit_norm_quant` | `csrc/kernels/patch_embed`, `csrc/conv`, `csrc/kernels/dit_bf16` | PyTorch eager and FlashRT reference | Later package |

## Naming Policy

Use names that describe math and data movement. Avoid model names in package and
API names unless the function cannot be made generic.

Good:

- `fp8_linear_bias_gelu`
- `residual_rmsnorm_quant`
- `smallm_nvfp4_gemm`
- `patch_embed_bias_pos`

Avoid:

- `qwen36_*`
- `pi05_*`
- `groot_*`
- `motus_*`

Model names may appear in benchmark shape labels or documentation explaining
provenance.
