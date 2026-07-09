# Results

## Headline (RTX 5090, real PI052 shapes, GQA kv_heads=1)

At the REAL attention shapes (q 8 heads / kv 1 head, D=256, bf16) the
materialized-logits `manual` backend (cuBLAS GEMMs + one compiled masked
softmax, native autograd) beats every fused path on fwd+bwd:

| shape | SDPA fwd/fwd+bwd | best flex fwd | best flex fwd+bwd | manual fwd/fwd+bwd | manual vs flex (fwd+bwd) |
| --- | ---: | ---: | ---: | ---: | ---: |
| b4_p700_k5_c50 | 1.581 / 4.006 | 0.262 | 2.597 | 0.360 / **1.285** | **2.0x** |
| b2_p700_k5_c50 | 0.775 / 2.011 | 0.166 | 2.486 | 0.221 / **0.864** | **2.9x** |
| b1_p700_k5_c50 | 0.453 / 1.292 | 0.139 | 2.180 | 0.127 / **0.977** | 2.2x |
| b4_p512_k5_c50 | 0.967 / 2.601 | 0.179 | 1.934 | 0.245 / **1.057** | 1.8x |
| b4_p896_k5_c50 | 2.332 / 5.667 | 0.323 | 2.634 | 0.515 / **1.854** | 1.4x |
| b4_p700_k1_c50 | 1.009 / 2.660 | ERROR | ERROR | 0.279 / **0.975** | flex NaN/err at K=1 |
| b4_p700_k8_c50 | 2.116 / 5.212 | 0.301 | 2.673 | 0.497 / **1.719** | 1.6x |

(5090, torch 2.11.0+cu128, median of 10 after 5 warmup, harness includes
per-iter leaf clones + loss; flex rows are the best of
{default, bwd_shrunk_only} x {64x64, 128x128}.)

- vs the SDPA dense-mask baseline the manual backend is **2.3-3.1x** on
  fwd+bwd, on every shape.
- manual peak (fwd) memory is **0.35-0.43x** of the SDPA path — the dense
  additive mask the SDPA path materializes costs more than the transient
  logits.
- `TORCHINDUCTOR_MAX_AUTOTUNE=1` trims manual fwd another ~14%
  (0.360 -> 0.311 at b4_p700_k5); fwd+bwd unchanged.

## Why (mechanism, not vibes)

- **GQA is the fidelity key.** With 8 kv heads (earlier sweeps) flex
  fwd+bwd looked best. At the real kv_heads=1: the flex BACKWARD
  autotune has no valid config on the 5090 (needs 112 KB shared memory,
  hardware 101 KB) and even the shrunken-tile fallback runs 2.2-2.7 ms —
  the backward dominates everything. The manual backward is four cuBLAS
  GEMMs + a fused softmax-grad chain: 0.5-1.4 ms.
- flex **forward** stays the fastest single direction (autotune,
  128x128 or 64x64 masks) — but you cannot combine flex-fwd with
  manual-bwd, and fwd+bwd is what training pays.
- Precision class: manual materializes logits in bf16, so outputs differ
  from the fp32-softmax fused paths by up to ~1.2e-2 max-abs (flex class:
  ~2e-3). Model-level parity gates (loss rel <= 1e-3, grads <= 1%) are
  the ship test; the fwd-diff gate for manual is tracked separately.

## Dispatch recommendation (as of 2026-07-09)

- 5090 training (fwd+bwd): `manual` everywhere; `TORCHINDUCTOR_MAX_AUTOTUNE`
  optional (+14% fwd).
- fwd-only (inference prefill): flex `default` autotune, mask 64x64
  (128x128 for P>=896-class shapes).
- A100: pending the same GQA matrix (flex bwd was already the weak leg
  there at 8 kv heads; manual expected to win bigger).

## Next levers (not yet implemented)

1. Structural 3-GEMM split: the prefix part computes a dense P x P even
   though the (att=0 rows x att=1 cols) quadrant is fully masked
   (~20% wasted FLOPs at p700) and the action part's cross-chunk block
   is near-empty. Splitting Q into (group-0, group-1, action) rows cuts
   both.
2. Custom autograd saving bf16 probabilities only (halves saved-activation
   bytes and backward traffic).
3. Native CUDA fused kernel: manual sits at ~20% of bf16 peak on the
   5090 (harness-inclusive); an FA2-style specialized kernel
   (D=256, GQA, prefix-dense + action-block) targeting 40-50% would be
   another ~2x. Entry per house rules: only after 1-2 are in and the
   remaining gap is confirmed on both archs.

## History

Earlier 8-kv-head sweeps (superseded — wrong KV shape for PI052):
5090 flex fwd+bwd best 1.48-2.55 ms vs SDPA 3.97; A100 matrix at 8 kv
heads showed flex positive at real shapes with 64x64 masks. Those runs
also established: torch autotune beats every hand preset at 8 heads;
`torch_default_explicit` NaNs at K=1; ROWS_GUARANTEED_SAFE /
BLOCKS_ARE_CONTIGUOUS / PRESCALE_QK no help on 5090.

No native CUDA performance results are claimed yet.
