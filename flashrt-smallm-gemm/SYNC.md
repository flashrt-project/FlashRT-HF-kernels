# Source Sync Plan

Upstream source: `../official/FlashRT`

Candidate source areas:

- `csrc/gemm/fp8_smallM_handtuned*`
- `csrc/kernels/bf16_matvec_*`
- `csrc/kernels/bf16_matmul_*`
- `csrc/kernels/fp4_w4a4_*`

## First Source Slice

Start with one kernel whose shape constraints can be documented cleanly without
model-specific terminology.
