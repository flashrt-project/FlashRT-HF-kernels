# Validation

Local source validation on NVIDIA GeForce RTX 5090:

```bash
python fp4-fused-ops/tests/test_fp4_fused_ops.py \
  --backend source \
  --mode full \
  --json-out internal-tests/fp4-fused-ops-source-full.json
```

Result:

- The current full source gate passed.
- Unsupported dimensions are rejected explicitly.
- Residual in-place updates are checked against the FP16 math contract.
- FP4/SFA outputs are dequantized and checked against the documented NVFP4
  quantization envelope.
- Linear NVFP4 pack/scale bytes use a bit-level reference.
- NCDHW RMSNorm, RMSNorm-SiLU and causal-cache outputs are checked against
  PyTorch and raw native launchers; fullgraph compile parity is covered.
- RTX 5090 SM120 full gate: `46/46` checks passed.
- Jetson AGX Thor SM110 model-shape gate: `58/58` checks passed across PI0.5,
  GROOT, Cosmos Edge, and LingBot VLA rows.

Representative correctness envelope from the full run:

| Workload | Shape | Max abs | Mean abs | P99 abs | Cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| residual+rms+FP4 v2 vs math reference | rows=1, dim=1024 | 0.453125 | 0.069949 | 0.306641 | 0.995567 |
| residual+rms+FP4 v2 vs math reference | rows=10, dim=2048 | 0.552734 | 0.071892 | 0.300781 | 0.995432 |
| residual+rms+FP4 v2 vs math reference | rows=64, dim=2048 | 0.589844 | 0.071508 | 0.303993 | 0.995464 |
| residual+rms+FP4 v2 vs math reference | rows=128, dim=4096 | 0.562500 | 0.071550 | 0.303711 | 0.995468 |
| silu-mul FP4 v2 vs v1 dequant | rows=10, dim=2048 | 0.023438 | 0.000022 | 0.000000 | 0.999895 |
| silu-mul FP4 v2 vs v1 dequant | rows=128, dim=4096 | 0.054688 | 0.000014 | 0.000000 | 0.999937 |

Notes:

- The residual/RMS rows compare a dequantized FP4 result to the FP16 math
  reference, so the nonzero error is expected NVFP4 quantization error.
- `packed_equal=False` can appear for v2-vs-v1 checks when the dequantized
  values are equivalent within the FP4 envelope; public validation is based on
  dequantized values plus residual contract, not byte identity alone.
## HF Jobs Publish Status

`flashrt/fp4-fused-ops` v1 was built and uploaded through the repository HF
Jobs workflow.

- Hub revision checked on June 20, 2026: `c77ac5a1`
- Uploaded variants:
  - `torch211-cxx11-cu128-x86_64-linux`
  - `torch211-cxx11-cu130-x86_64-linux`
  - `torch212-cxx11-cu130-x86_64-linux`
  - `torch212-cxx11-cu132-x86_64-linux`

The SM110 `torch211-cxx11-cu130-aarch64-linux` artifact and cold Hub load are
required release gates before claiming published Thor support.

## Thor Native Parity

The package compile flags match the native FlashRT target, including
`--use_fast_math`. Without that flag, gated producers regress materially even
though correctness passes. With the release flags, CUDA Graph wrapper/native
latency over 30 production-model rows had median `1.001`, p95 `1.052`, and
maximum `1.082` on Thor.
