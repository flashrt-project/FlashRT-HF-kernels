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

## Installed artifact on NVIDIA Thor (SM110)

The installed-artifact gate (missing when this card was written) is now closed
on Thor `sm_110a`. Build notes:

- Requires CUTLASS **4.4.0** (the package declares `cutlass_4_4`); 4.5.x moved
  `SM100_MMA_F8F6F4_SS` to a class template and breaks `csrc/fmha77`.
- `cutlass/util/packed_stride.hpp` is vendored by `fp4-gemm` and must be on the
  include path (same local-build workaround as `fp8-gemm`).
- Compiled with `-gencode arch=compute_110a` and the FMHA sm100 kernel runs
  natively on Thor (no fallback required, unlike `fused-mlp-megakernels`).

Command:

```bash
python fp8-cross-attention-blackwell/tests/test_fp8_cross_attention_blackwell.py \
    --backend installed --artifact <installed-dir> --mode full
```

Result: 9/9 numeric rows, invalid-head rejection, CUDA Graph replay, and
`torch.compile(fullgraph=True)` all passed on SM110. Worst row across the
matrix (`B1,Sq786,Sk7984,Hq28,Hkv4,D128`): `max=0.00025749`, `p99=0.00012207`,
`mean=0.00003710`, `cosine=0.99978602` — comfortably inside the
`max<=0.004 / cosine>=0.9995` gate. Other rows held `cosine>=0.999778`.
