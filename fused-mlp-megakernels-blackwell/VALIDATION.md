# Validation

Release gates require:

1. Source and installed-artifact correctness against the FP16 PyTorch formula.
2. Max, p99, mean absolute error, cosine similarity, dtype, and shape checks.
3. Boundary and production shapes, including non-tile-aligned M and the
   PI-style encoder row `M=768,N=16384,K=2048`.
4. `torch.compile(fullgraph=True)` and prewarmed CUDA Graph replay.
5. Benchmark comparison against PyTorch eager, `torch.compile`, and the native
   FlashRT entry on the same hardware and stream.
6. Per-architecture artifact execution before a hardware claim is published.

NVIDIA Thor SM110 source validation passed seven rows from `128x128x128`
through `768x16384x2048`, including M-tail rows 127 and 129, plus
`torch.compile(fullgraph=True)` and CUDA Graph replay. The production row
recorded `max=0.00146484`, `p99=0.00012207`, `mean=0.00001121`, and
`cosine=0.99999988`. Installed-artifact execution remains a separate release
gate.
