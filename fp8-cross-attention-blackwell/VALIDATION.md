# Validation

Release requires source and installed-artifact tests on each claimed
architecture. The matrix covers self/cross attention, MHA/GQA, unequal and
non-tile-aligned sequence lengths, batch 1/2, and the production
`B1,Sq786,Sk7984,Hq28,Hkv4,D128` row. Every row records max/p99/mean error,
cosine, and dtype against dequantized PyTorch SDPA. CUDA Graph replay and
native FlashRT latency parity are mandatory.

## SM110 source result

NVIDIA Thor, CUDA 13, PyTorch 2.11.0+cu130 passed nine numeric rows covering
sequence boundaries 1, 127, 128, 129, 255, 256, 257, and 513, plus the
production row above. GQA groups 1, 4, and 8, batch sizes 1 and 2, CUDA Graph
replay, and invalid head divisibility were covered.

The numeric gate is `max<=0.004`, `p99<=0.002`, `mean<=0.0005`, and
`cosine>=0.9995`. The final partial KV tile uses `ResidualMask`; regressions on
both sides of every 128-token boundary are release blockers.

Installed-artifact correctness and `torch.compile(fullgraph=True)` remain
mandatory after the HF Jobs build. Source validation does not replace them.
