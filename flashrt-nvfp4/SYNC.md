# Source Sync Plan

Upstream source: `../official/FlashRT`

Candidate source areas:

- `csrc/quantize/quantize_fp4_dynamic.*`
- `csrc/quantize/quantize_fp4_sfa.*`
- `csrc/quantize/reshape_scales_sfa.*`
- `csrc/fused_fp4/`
- selected headers from `csrc/gemm/fp4/`

## First Source Slice

Recommended first APIs:

```text
quantize_nvfp4_sfa(input, is_weight: bool) -> (packed, sfa)
reshape_linear_scales_to_sfa(scales, rows, dim, is_weight: bool) -> Tensor
sfa_size_bytes(rows, dim, is_weight: bool) -> int
```
