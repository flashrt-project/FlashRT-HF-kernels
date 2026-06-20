# fp4-gemm Examples

These examples show direct Hub-style usage of `flashrt/fp4-gemm`.

```bash
python fp4-gemm/examples/fp4_gemm_linear.py
```

The quantization helper is included for validation and small examples. In a
runtime, weights should normally be prepacked and loaded as FP4/SFA/SFB buffers.
