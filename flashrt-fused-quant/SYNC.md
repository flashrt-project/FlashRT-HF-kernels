# Source Sync Plan

Upstream source: `../official/FlashRT`

Candidate source areas:

- `csrc/kernels/norm.*`
- `csrc/kernels/fusion.*`
- `csrc/kernels/quantize.*`
- `csrc/kernels/rope.*`
- `csrc/quantize/qkv_split_norm_rope_bf16.*`
- selected fused FP4 files only when the API is not layout-only

## First Source Slice

Recommended first API:

```text
residual_rmsnorm_quant(input, residual, weight, scale, dtype) -> Tensor
```

Use a PyTorch reference implementation for correctness.
