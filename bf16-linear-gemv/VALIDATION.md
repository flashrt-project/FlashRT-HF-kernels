# Validation

Required source gate:

```bash
python bf16-linear-gemv/tests/test_bf16_linear_gemv.py --backend source --mode full
```

Correctness metrics:

- max absolute error
- mean absolute error
- cosine similarity

Reference: PyTorch `x.float() @ weight.float().T` cast to BF16.

This package is source-ready for HF Jobs. Built-artifact and multi-hardware
rows should be added after the release build.
