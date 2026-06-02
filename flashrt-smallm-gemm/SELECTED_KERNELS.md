# Selected Kernels

`flashrt-smallm-gemm` targets decode and low-latency small-M inference shapes.
This is a community gap: generic GEMM libraries are excellent at large
throughput problems but often leave launch and tiling overhead on tiny-M shapes.

## Tier 1: NVFP4 W4A4 Decode

| Public API | FlashRT source | Reason |
| --- | --- | --- |
| `nvfp4_w4a4_decode_matvec_bf16out` | `official/FlashRT/csrc/kernels/fp4_w4a4_matvec_sm120.cu` | M=1 decode path with low launch overhead and direct BF16 output. |
| `nvfp4_w4a4_smallm_warpsplit_bf16out` | `official/FlashRT/csrc/kernels/fp4_w4a4_mma_warpsplit_mrows_sm120.cu` | Small-M MMA path that splits K work across warps for decode-like batches. |

## Tier 2: Tiny FP8 Small-M

| Public API | FlashRT source | Reason |
| --- | --- | --- |
| `tiny_fp8_smallm_gemm_bf16out` | `official/FlashRT/csrc/kernels/megakernel/tinyfp8_kernels_sm120.cu` | Hand-specialized FP8 GEMM families for small fixed M/N/K shapes. |

## Shape Policy

Do not hide shape specialization. Public docs should list the exact dispatch
families and provide a fallback recommendation when a shape is outside the
tuned grid.

Initial benchmark grid:

- `M in {1, 2, 4, 8, 16, 32}`.
- Decode-like K/N pairs from Transformer and VLA blocks.
- Tiny FP8 fixed families exposed in the upstream binding table.

## Promotion Rule

This package should not promote a generic `smallm_gemm` name until the dispatch
wrapper chooses among measured tile policies. Before that, export exact
dtype/layout/shape-family APIs.
