# Source Sync Plan

Upstream source: `../official/FlashRT`

Selected source areas:

| Package path | Upstream path | Status |
| --- | --- | --- |
| `csrc/fp4_w4a4_matvec_sm120.*` | `official/FlashRT/csrc/kernels/fp4_w4a4_matvec_sm120.*` | Synced |
| `csrc/fp4_w4a4_mma_warpsplit_mrows_sm120.*` | `official/FlashRT/csrc/kernels/fp4_w4a4_mma_warpsplit_mrows_sm120.*` | Draft target |
| `csrc/tinyfp8_kernels_sm120.*` | `official/FlashRT/csrc/kernels/megakernel/tinyfp8_kernels_sm120.*` | Draft target |

## First Source Slice

The first synced slice is W4A4 matvec because the shape constraints are easiest
to document without model-specific terminology. Add warpsplit and tiny FP8 only
after the matvec Tensor binding and reference test are stable.
