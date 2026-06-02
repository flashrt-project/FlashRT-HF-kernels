# Source Sync Plan

Upstream source: `../official/FlashRT`

Selected source areas:

| Package path | Upstream path | Status |
| --- | --- | --- |
| `csrc/silu_mul_to_nvfp4_swizzled.*` | `official/FlashRT/csrc/kernels/silu_mul_to_nvfp4_swizzled.*` | First sync target |
| `csrc/norm_silu_fp4_sfa.*` | `official/FlashRT/csrc/fused_fp4/norm_silu_fp4_sfa.*` | Draft target |
| `csrc/res_rms_fp4_sfa_v2.*` | `official/FlashRT/csrc/fused_fp4/res_rms_fp4_sfa_v2.*` | Draft target |
| `csrc/bagel_res_rms_fp4_sfa_bf16.cu` | `official/FlashRT/flash_wm/csrc/bagel_res_rms_fp4_sfa_bf16.cu` | Draft target with generic public name |
| `csrc/bagel_silu_mul_fp4_sfa.cu` | `official/FlashRT/flash_wm/csrc/bagel_silu_mul_fp4_sfa.cu` | Draft target with generic public name |

## First Source Slice

Recommended first APIs:

```text
silu_mul_quant_nvfp4_swizzled_bf16(gate, up) -> (packed, scales)
silu_mul_merged_quant_nvfp4_swizzled_bf16(merged) -> (packed, scales)
```

Use a PyTorch fake-quant reference implementation for correctness. Add
residual/RMSNorm variants after the activation+quant path is stable.
