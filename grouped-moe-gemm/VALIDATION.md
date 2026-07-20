# Validation

Each release compares all three dispatch paths against an explicit FP32
dequantize-then-matmul reference and reports max, p99, mean absolute error,
cosine, output dtype and tolerance. Gates include multiple experts, repeated
experts, sentinel tiles, K/N boundaries, CUDA Graph, fullgraph compile and
installed-artifact cold load. Performance is compared to the original FlashRT
entry points and relevant eager/compiled references without counting Python
packing loops as a kernel speedup baseline.
