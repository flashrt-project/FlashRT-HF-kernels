# Validation

Release gates compare each fused region with an operation-by-operation reference
built from the exact quantized inputs, BF16 rounding points and static scales.
Every supported M boundary reports max/p99/mean absolute error, cosine, dtype
and tolerance. The matrix includes CUDA Graph reuse, fullgraph tracing, invalid
contracts, original FlashRT parity and built-artifact cold load.

Benchmarks compare the full fused region against equivalent eager and compiled
regions and exclude one-time weight preparation and buffer allocation.

The SM110 installed-artifact matrix covers gated `M=1,8,21,32` and residual
`M=1,51,144,188`, plus fullgraph tracing and CUDA Graph replay. FP8 tensor-core
accumulation is checked with max, p99, mean, cosine, dtype, and per-shape
tolerances. Every SM110 gated and residual row is also replayed with identical
static inputs and scratch buffers and must be bitwise deterministic. Five
independent full-matrix processes passed after the SM110 shared-stage race fix.
