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

Composite fwd+bwd with persistent leaves (no per-iteration weight clone),
kernel vs plain vs compiled-plain:

| N | kernel | plain | compiled | target |
|---|---|---|---|---|
| 46 | 5.65 | 5.43 | 4.88 | <= 3.9 |
| 126 | 11.88 | 8.99 | 7.82 | <= 6.0 |
| 504 (fallback) | 32.04 | 31.97 | 27.07 | no-regress: PASS |

Isolated fwd kernel journey: 10.6 -> 4.16 -> **2.32 ms (930 GB/s)** at
N=46 (N=126: 9.17 -> 6.13):
1. exact-JN templating — a variable-bound loop over a locally indexed
   accumulator array spilled to local memory (11% of bandwidth);
2. float4 shared-memory reads (16B row padding);
3. cp.async double-buffered staging + N-adaptive H-chunk (64 for N<=64,
   32 above) keeping two CTAs per SM where the tiles fit.

Measured negative: deeper pipelines with smaller chunks (3-4 stages x
32-float) are SLOWER than 2 stages x 64 (2.56/6.49 vs 2.32/6.13) — sync
and commit overhead beats the extra overlap at these tile sizes. Also
fixed en route: the smem-attribute latch (configure with the kMaxRows
size, or a larger-N launch silently fails after a smaller-N first call)
+ C10_CUDA_KERNEL_LAUNCH_CHECK in the binding.

## Sober gate re-assessment (clean baselines)

- N=46: plain/cuBLAS is already near floor (sum 5.33 with GEMMs at
  62-88%). Our theoretical best ~= 4.4-4.8 ms (fwd ~1.4 + dX 1.33 +
  fused-dW ~1.4 + merge) = 1.1-1.2x over plain — BELOW the 1.3x ship
  gate. The tiny-N case is likely not winnable by the required margin;
  proposal: narrow the primary target to N in [64, 128].
- N=126: reachable (~6.0-6.3 vs target 6.0) but needs three more pieces:
  fwd retile (kVTile=64: halves smem reads per FMA and fixes the
  1-CTA/SM occupancy at N>64), a custom dX pass (cuBLAS sits at ~50%
  there too), and the fused-dW backward. Estimated 2-3 focused
  iterations.

Not wired into any model path until Gate 2 passes.
