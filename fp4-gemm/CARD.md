# flashrt/fp4-gemm

FlashRT native Blackwell NVFP4 A4W4 GEMM kernels. Both activations and weights
are packed FP4 inputs; this is not a BF16-activation weight-only operation.

## Functions

- `sfa_size_bytes`
- `capabilities`
- `quantize_fp4_sfa_fp16`
- `quantize_fp4_sfa_bf16`
- `quantize_fp4_sfa_mse_fp16`
- `quantize_fp4_sfa_mse_bf16`
- `quantize_fp4_sfa_padded_bf16`
- `pack_nvfp4_weight_bf16`
- `quantize_e0m3_sfa_fp16`
- `dequantize_fp4_sfa_fp16`
- `nvfp4_gemm_bf16`
- `nvfp4_gemm_fp16`
- `nvfp4_gemm_variant_bf16`
- `nvfp4_gemm_nvfp4`
- `nvfp4_gemm_geglu_nvfp4_fp16`
- `cutlass_fp4_gemm_geglu_il_hw_v10`
- `nvfp4_gemm_bias_gelu_nvfp4_fp16`
- `nvfp4_gemm_bias_residual_fp16`
- `nvfp4_gemm_bias_bf16`
- `nvfp4_gemm_bias_residual_bf16`
- `nvfp4_gemm_residual_bf16`
- `nvfp4_gemm_bias_gelu_bf16`
- `nvfp4_gemm_bias_gelu_nvfp4`
- `nvfp4_gemm_streamk_bf16`
- `nvfp4_gemm_streamk_bias_bf16`
- `fp4_w4a16_linear_bf16` (compatibility alias)
- `e0m3_weight_gemm_fp16`
- `nvfp4_gemm_relu2_nvfp4`

## Example

```python
from kernels import get_kernel
import torch

ops = get_kernel("flashrt/fp4-gemm", version=1, trust_remote_code=True)

x = torch.randn((32, 256), device="cuda", dtype=torch.float16)
w = torch.randn((512, 256), device="cuda", dtype=torch.float16)

a, sfa = ops.quantize_fp4_sfa_fp16(x, is_sfb=False)
b, sfb = ops.quantize_fp4_sfa_fp16(w, is_sfb=True)
y = ops.nvfp4_gemm_bf16(a, b, sfa, sfb)
```

BF16 activations should use the direct producer to avoid a separate cast and
copy before every low-bit projection:

```python
x = torch.randn((1, 5120), device="cuda", dtype=torch.bfloat16)
a, sfa = ops.quantize_fp4_sfa_bf16(x)
```

## Notes

- Blackwell `sm_110a` with CUDA 13+ and `sm_120a` with CUDA 12.8+.
- Inputs are packed FP4 E2M1 plus CUTLASS Sm1xx SFA/SFB scale buffers.
- Output is BF16.
- Read `capabilities()` instead of duplicating scale-factor layout or alignment
  constants in a runtime integration. Unsupported calls raise; output is never
  silently left undefined.
- `variant=-1` is the architecture-aware production auto-dispatch;
  `variant=0/1/2` expose diagnostic default, widen, and pingpong schedules.
- The canonical BF16-output GEMM, fused bias GEMM, and FP4 pack/unpack helpers
  support SM110 and SM120. SM110 also supports bias+residual and bias+GELU-to-FP4
  production epilogues used by the GROOT N1.7 Thor pipeline.
- Dimensions used by Blackwell NVFP4 TMA are physically aligned to 32. For
  logical widths such as SigLIP 4304, use the bind-time padding/packing helpers
  to create static 4320 tensors; the GEMM hot path performs no padding.
