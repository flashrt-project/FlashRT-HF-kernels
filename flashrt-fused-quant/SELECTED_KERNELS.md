# Selected Kernels

`flashrt-fused-quant` is the shared memory-bound fusion package. It should
cover operation chains that users otherwise express as multiple PyTorch launches
around activation, normalization, residual updates, and low-bit quantization.

## Tier 1: Activation + NVFP4 Quant

| Public API | FlashRT source | Reason |
| --- | --- | --- |
| `silu_mul_quant_nvfp4_swizzled_bf16` | `official/FlashRT/csrc/kernels/silu_mul_to_nvfp4_swizzled.cu` | Fuses `SiLU(gate) * up` with BF16-to-NVFP4 swizzled quantization. |
| `silu_mul_merged_quant_nvfp4_swizzled_bf16` | `official/FlashRT/csrc/kernels/silu_mul_to_nvfp4_swizzled.cu` | Same math for merged `[S, 2H]` gate/up tensors, which is common in Transformer FFN blocks. |

## Tier 2: Residual/RMSNorm + Low-Bit Quant

| Public API | FlashRT source | Reason |
| --- | --- | --- |
| `rmsnorm_quant_nvfp4_sfa_fp16` | `official/FlashRT/csrc/fused_fp4/norm_silu_fp4_sfa.cu` | Fuses RMSNorm and FP4/SFA output for layer-entry paths. |
| `residual_rmsnorm_quant_nvfp4_sfa_fp16` | `official/FlashRT/csrc/fused_fp4/res_rms_fp4_sfa_v2.cu` | Residual update, RMSNorm, and low-bit quant in one pass. |
| `residual_rmsnorm_quant_nvfp4_sfa_bf16` | `official/FlashRT/flash_wm/csrc/bagel_res_rms_fp4_sfa_bf16.cu` | BF16 input/output-residual-safe variant for model paths that cannot use FP16 intermediates. |

## Tier 3: True SiLU + SFA Variant

| Public API | FlashRT source | Reason |
| --- | --- | --- |
| `silu_mul_quant_nvfp4_sfa_fp16` | `official/FlashRT/flash_wm/csrc/bagel_silu_mul_fp4_sfa.cu` | True-SiLU activation fused with FP4/SFA quantization; useful when upstream naming or activation approximations differ. |

## Validation Grid

- Hidden sizes from Transformer and VLA FFN blocks.
- Sequence/token counts from decode, prefill, image tokens, and video tokens.
- BF16/FP16 input variants documented separately.
- Baselines are PyTorch eager chains plus FlashRT internal parity.

## Promotion Rule

This package should graduate after `flashrt-nvfp4` layout helpers are stable,
because several outputs need the same public explanation of FP4/NVFP4 scale
factor layout.
