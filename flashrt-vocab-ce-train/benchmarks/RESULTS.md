# flashrt-vocab-ce-train — status: fwd kernel WIP (Gate 1 PASS, Gate 2 open)

Environment: RTX 5090, torch 2.11.0+cu128. V=257152, H=2048, fp32 head.

## Correctness (Gate 1): PASS

Streaming fwd kernel + torch merge/loss + torch backward vs the
materialized-logits reference: loss bitwise-identical on most cases
(<= 2e-6 rel with z-loss), grads <= 2e-6 rel fp32 / 3e-3 bf16-hidden;
all-ignored returns exact 0.0 sync-free; N>128 dispatches to the plain
path bit-identically; fp64 gradcheck of the reference passes.

## Performance (Gate 2): OPEN — optimization ladder in progress

Composite fwd+bwd (kernel path vs plain vs compiled-plain, same harness):

| N | kernel | plain | compiled | target |
|---|---|---|---|---|
| 46 | 10.41 | 8.01 | 7.57 | <= 3.9 |
| 126 | 17.11 | 11.16 | 9.82 | <= 6.0 |
| 504 (fallback) | 34.21 | 34.11 | 29.01 | no-regress: PASS |

Isolated fwd kernel: N=46 4.16 ms (518 GB/s), N=126 9.17 ms. Journey so
far: 10.6 -> 4.16 via (a) exact-JN templating — a variable-bound loop over
a locally indexed accumulator array spilled to local memory (11% of
bandwidth), (b) float4 shared-memory reads (16B row padding).

Known bottlenecks, next in line:
1. Staging is now dominant (staging-only variant measures 3.12 ms at 675
   GB/s): synchronous smem loads with two barriers per H-chunk expose full
   memory latency — cp.async double-buffering is the designed fix.
2. N>64 drops to 1 CTA/SM (h tile 66+ KB smem): make kHChunk adaptive in N
   to hold >= 2 CTAs/SM.
Path to the targets: max(staging, compute) overlap ~1.3-1.5 ms fwd at
N=46, plus the fused-dW backward pass from the spec.

Not wired into any model path until Gate 2 passes.
