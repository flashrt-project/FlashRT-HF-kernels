# flashrt-vocab-ce-train — status: Gate 1 PASS, Gate 2 CLOSED (not shippable)

Environment: RTX 5090, torch 2.11.0+cu128. V=257152, H=2048, fp32 head.

## Verdict

The custom-kernel path cannot clear the ship gate
(`min(plain/1.3, compiled/1.2, spec target)`) at these shapes: the three
fp32 GEMMs dominate the region (6.9-7.3 ms of the ~8 ms composite at
N=126) and cuBLAS runs each of them at 47-89% of DRAM peak — every
streaming variant we built measured BELOW cuBLAS on the same GEMM
(best 595 GB/s vs cuBLAS 955 GB/s at N=126). Beating the gate would
require a from-scratch register-blocked fp32 GEMM that outruns cuBLAS by
~1.3x on three different shapes; that is a separate project, not an
epilogue-fusion win. Per the acceptance protocol the shipped change is
the compiled plain path; this package stays unwired.

## Correctness (Gate 1): PASS (hybrid path)

Hybrid forward (cuBLAS logits GEMM + fused stats kernel) + torch
merge/loss + torch backward vs the materialized-logits reference: loss
bitwise-identical on most cases (<= 2e-6 rel with z-loss), grads <= 2e-6
rel fp32 / 3e-3 bf16-hidden; all-ignored returns exact 0.0 sync-free;
N>128 dispatches to the plain path bit-identically; fp64 gradcheck of
the reference passes.

## Performance (Gate 2): CLOSED

Composite fwd+bwd, persistent-leaf harness (kernel path vs plain vs
compiled-plain):

| N | kernel (final hybrid) | plain | compiled | gate | verdict |
|---|---|---|---|---|---|
| 46 | 5.08 | 5.37 | 4.94 | <= 3.9 | FAIL (beats plain, loses to compiled) |
| 126 | 8.27 | 8.98 | 7.81 | <= 6.0 | FAIL (beats plain, loses to compiled) |
| 504 (fallback) | 32.0 | 32.0 | 27.0 | no-regress | PASS |

Kernel-path journey at N=126: 17.1 -> 11.9 (fwd streaming ladder) ->
10.1 (kVTile retile) -> **8.27 (hybrid: cuBLAS GEMM + fused stats)**.

### Phase split that decides the verdict (measured, this HEAD)

| phase | N=46 | N=126 | cuBLAS % of DRAM peak |
|---|---|---|---|
| logits GEMM (h @ W.t) | 1.83 | 2.27 | 53% (and best of all transpose variants) |
| dX GEMM (best: dlT.t @ W) | 1.39 | 2.28 | 47-89% |
| dW GEMM (dl.t @ h) | 1.53 | 2.41 | 48% |
| CE+lse torch chain | 0.15 | 0.80 | replaced by stats kernel: ~0.12 |
| dlogits torch chain | 0.11 | 0.53 | fusable to ~0.2 |

GEMMs alone sum to 6.96 ms at N=126 with the best layouts; the gate is
6.0. All non-GEMM fusion upside combined (~1.1 ms) cannot close it.

## What the package now contains

- `vocab_ce_fwd_stream`: the streaming fwd kernel (10.6 -> 2.32 ms /
  930 GB/s at N=46 over the optimization ladder; kVTile=64 retile takes
  N=126 from 6.13 to 4.5 ms). Kept for reference — cuBLAS still wins the
  same GEMM (1.83 / 2.27 ms).
- `vocab_ce_stats`: one-pass online-softmax partials over materialized
  logits; replaces the separate CE + logsumexp torch passes
  (0.8 -> ~0.12 ms at N=126). This piece works and is reusable.
- Hybrid autograd path wiring both, Gate 1 green.

## Lessons captured (do not re-derive)

1. Exact-JN templating: variable-bound loops over locally indexed
   accumulators spill to local memory (11% of bandwidth).
2. float4 smem reads with 16B row pad; h-broadcast reads need no pad.
3. cp.async 2-stage double buffer; deeper pipelines measured slower.
4. smem-attribute latch: configure `cudaFuncSetAttribute` with the max
   size or larger-N launches silently fail after a smaller-N first call.
5. Persistent-leaf bench harness (weight clones add ~2.5 ms to every path).
6. Standalone-probe buffers must hold non-constant data: memset-zero
   weights hit DRAM compression on Blackwell and inflate bandwidth ~40%.
7. Occupancy sweep at N=126: t512/kVPer=2/hChunk=32 optimum; 768-1024
   threads, 16-float chunks, 4 vocab rows/thread, and L2-direct hidden
   reads (no smem staging) all slower.
8. GEMM transpose variants: dX prefers a pre-transposed (V,N) dlogits
   operand (2.89 -> 2.28 ms); fwd and dW are already optimal as written.
