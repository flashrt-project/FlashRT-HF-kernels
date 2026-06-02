# Source Directory

Selected first source slices:

```text
official/FlashRT/csrc/kernels/silu_mul_to_nvfp4_swizzled.cu
official/FlashRT/csrc/kernels/silu_mul_to_nvfp4_swizzled.cuh
official/FlashRT/csrc/fused_fp4/norm_silu_fp4_sfa.cu
official/FlashRT/csrc/fused_fp4/norm_silu_fp4_sfa.cuh
official/FlashRT/csrc/fused_fp4/res_rms_fp4_sfa_v2.cu
official/FlashRT/flash_wm/csrc/bagel_res_rms_fp4_sfa_bf16.cu
official/FlashRT/flash_wm/csrc/bagel_silu_mul_fp4_sfa.cu
```

Public binding names should be generic even when a source file came from a
model-specific FlashRT path.
