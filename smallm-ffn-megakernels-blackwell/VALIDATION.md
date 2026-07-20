# Validation

Release gates compare each fused region with an operation-by-operation reference
built from the exact quantized inputs, BF16 rounding points and static scales.
Every supported M boundary reports max/p99/mean absolute error, cosine, dtype
and tolerance. The matrix includes CUDA Graph reuse, fullgraph tracing, invalid
contracts, original FlashRT parity and built-artifact cold load.

Benchmarks compare the full fused region against equivalent eager and compiled
regions and exclude one-time weight preparation and buffer allocation.
