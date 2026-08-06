# flashrt/fp4-fused-ops

FlashRT fused FP16-to-NVFP4 producer kernels for transformer and diffuser
low-bit paths.

## Functions

- `sfa_size_bytes`
- `rms_norm_fp4_sfa_fp16`
- `residual_add_rms_norm_fp4_sfa_fp16`
- `residual_add_rms_norm_fp4_sfa_v2_fp16`
- `residual_add_rms_norm_mul_fp4_sfa_fp16`
- `silu_mul_fp4_sfa_fp16`
- `silu_mul_fp4_sfa_v2_fp16`
- `silu_mul_mul_fp4_sfa_v2_fp16`
- `silu_mul_two_fp4_to_fp4`
- `silu_mul_two_mul_fp4_to_fp4`
- `dequantize_fp4_sfa_fp16`
- `quantize_bf16_to_nvfp4_linear`
- `rms_silu_nvfp4_ndhwc_bf16`
- `bf16_rms_norm_ncdhw`
- `bf16_rms_silu_ncdhw`
- `adaptive_rms_norm_e0m3_fp16`
- `gated_residual_adaptive_rms_norm_e0m3_fp16`
- `gelu_mul_e0m3_fp16`
- `residual_add_rms_norm_quant_nvfp4_swizzled_bf16`
- `relu2_quant_nvfp4_swizzled_fp16`

This package targets Blackwell `sm_110a` and `sm_120a` and uses CUTLASS/CUTE
SFA layouts. SM110 artifacts require CUDA 13+.

## Example

```python
from kernels import get_kernel
import torch

ops = get_kernel("flashrt/fp4-fused-ops", version=1, trust_remote_code=True)

merged = torch.randn((16, 4096), device="cuda", dtype=torch.float16)
packed, sfa = ops.silu_mul_fp4_sfa_v2_fp16(merged)

# Debug only; normal low-bit pipelines should pass packed/SFA to FP4 GEMM.
bf16_view = ops.dequantize_fp4_sfa_fp16(packed, sfa)
```

## Shape Contract

- CUDA tensors only.
- FP16 producer inputs, uint8 FP4 packed outputs, uint8 CUTLASS SFA buffers.
- Dimensions must be divisible by 16.
- v1 RMS producer paths support `dim <= 2048`.
- Larger residual/RMS producer shapes should use
  `residual_add_rms_norm_fp4_sfa_v2_fp16`.
- NCDHW BF16 RMS kernels require even `C <= 1024`; the fused NCDHW-to-NDHWC
  NVFP4 producer requires `C % 128 == 0`.
