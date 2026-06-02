# Source Sync Plan

Upstream source: `../official/FlashRT`

Selected source areas:

| Package path | Upstream path | Status |
| --- | --- | --- |
| `csrc/nvfp4_sf_reshape_sm120.*` | `official/FlashRT/csrc/quantize/nvfp4_sf_reshape_sm120.*` | Synced, public comments sanitized |
| `csrc/cutlass_nvfp4_gemm_bias_gelu_fp4out_sm120.*` | `official/FlashRT/csrc/gemm/fp4/cutlass_nvfp4_gemm_bias_gelu_fp4out_sm120.*` | Draft target |
| `csrc/cutlass_nvfp4_gemm_bias_gelu_bf16out_sm120.*` | `official/FlashRT/csrc/gemm/fp4/cutlass_nvfp4_gemm_bias_gelu_bf16out_sm120.*` | Draft target |
| `csrc/cutlass_nvfp4_gemm_dn_streamk_bias_sm120.*` | `official/FlashRT/csrc/gemm/fp4/cutlass_nvfp4_gemm_dn_streamk_bias_sm120.*` | Draft target |

## First Source Slice

Implemented first APIs:

```text
nvfp4_sf_linear_to_swizzled(scales, rows, cols, is_weight: bool) -> Tensor
nvfp4_sf_swizzled_bytes(rows, cols, is_weight: bool) -> int
```

The fused GEMM epilogues should be synced after the layout helper is buildable,
because they bring in the heavier CUTLASS dependency surface.
