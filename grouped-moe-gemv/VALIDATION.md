# Validation

Required source gate:

```bash
python grouped-moe-gemv/tests/test_grouped_moe_gemv.py --backend source --mode full
```

The current source gate uses deterministic NVFP4 packed constants and validates
exact BF16 output against the analytically expected value. Full random packed
weight sweeps should be added before broad public performance claims.
