# fp4-fused-ops Examples

These examples show direct Hub-style usage of `flashrt/fp4-fused-ops`.

```bash
python fp4-fused-ops/examples/fp4_fused_ops_block.py
```

The hot path should pass packed FP4/SFA outputs directly to adjacent FP4 GEMM
kernels. The example dequantizes only to make the result easy to inspect.
