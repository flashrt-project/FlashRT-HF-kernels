# Validation

Each release compares all three dispatch paths against an explicit FP32
dequantize-then-matmul reference and reports max, p99, mean absolute error,
cosine, relative L2, output dtype and tolerance. Gates include multiple
experts, repeated experts, K/N boundaries, CUDA Graph, fullgraph compile and
installed-artifact cold load. The exact 35B gate/up and down projection shapes
are mandatory rows. Performance is compared to the original FlashRT
entry points and relevant eager/compiled references without counting Python
packing loops as a kernel speedup baseline.

```bash
python grouped-moe-gemm/tests/test_grouped_moe_gemm.py --backend source
python grouped-moe-gemm/tests/test_grouped_moe_gemm.py \
  --backend installed --artifact build/torch211-cxx11-cu128-x86_64-linux
python grouped-moe-gemm/benchmarks/benchmark.py --backend source
```
