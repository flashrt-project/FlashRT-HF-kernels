# Validation

Local source validation on NVIDIA GeForce RTX 5090:

```bash
python fp4-fused-ops/tests/test_fp4_fused_ops.py \
  --backend source \
  --mode full \
  --json-out internal-tests/fp4-fused-ops-source-full.json
```

Result:

- `26/26` checks passed.
- Unsupported dimensions are rejected explicitly.
- Residual in-place updates are checked against the FP16 math contract.
- FP4/SFA outputs are dequantized and checked against the documented NVFP4
  quantization envelope.

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
- Installed artifact validation is required after HF Jobs publishes the package.
