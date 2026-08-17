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

## Installed artifact on NVIDIA Thor (SM110) via portable SIMT fallback

The SM100-family CUTLASS 4.0 megakernel uses tcgen05/TMA descriptor paths that
assert at runtime on `sm_110a` (Thor). `portable_geglu_simt.cu` provides a pure
SIMT FMA reference of the same fusion:

- `gate_scratch[m,n] = fp16( gelu_tanh( X@W_gate^T ) )`
- `hidden[m,n]     = fp16( gate_scratch[m,n] * ( X@W_up^T ) )`

Dispatch in `torch_binding.cpp`: `sm_100`/`sm_103` keep the CUTLASS megakernel;
`sm_110a` and the `FLASHRT_FORCE_SIMT` override route to the SIMT reference
(mirrors the fp8-gemm / world-model-conv / grouped-moe-gemm fallbacks).

Command:

```bash
python fused-mlp-megakernels-blackwell/tests/test_fused_mlp_megakernels_blackwell.py \
    --backend installed --artifact <installed-dir> --mode full
```

Result: 7/7 numeric rows passed on Thor, plus `torch.compile(fullgraph=True)`
and prewarmed CUDA Graph replay. Rows and metrics (vs the PyTorch FP16 formula
`gelu_tanh(x@Wg.t()) * (x@Wu.t())`):

| M | N | K | max_abs | p99 | mean | cosine |
|---|---|---|---|---|---|---|
| 128 | 128 | 128 | 0.0000305 | 0.0000076 | 0.0000007 | 0.99999988 |
| 64 | 256 | 256 | 0.0000610 | 0.0000153 | 0.0000014 | 0.99999976 |
| 127 | 256 | 256 | 0.0000610 | 0.0000153 | 0.0000014 | 0.99999982 |
| 129 | 256 | 256 | 0.0000610 | 0.0000153 | 0.0000014 | 0.99999982 |
| 256 | 1024 | 1024 | 0.0004883 | 0.0000610 | 0.0000056 | 0.99999988 |
| 768 | 2048 | 2048 | 0.0009766 | 0.0001221 | 0.0000113 | 0.99999994 |
| 768 | 16384 | 2048 | 0.0014648 | 0.0001221 | 0.0000112 | 0.99999988 |

The SIMT path is a correctness/compatibility fallback for non-SM100-family
devices; `sm_100`/`sm_103` continue to use the fused CUTLASS megakernel.
Performance qualification against PyTorch eager / `torch.compile` remains an
open release gate (see item 5 above).
