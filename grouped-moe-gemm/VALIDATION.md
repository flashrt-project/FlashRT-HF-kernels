# Validation

Each release compares all three dispatch paths against an explicit FP32
dequantize-then-matmul reference and reports max, p99, mean absolute error,
cosine, output dtype and tolerance. Gates include multiple experts, repeated
experts, sentinel tiles, K/N boundaries, CUDA Graph, fullgraph compile and
installed-artifact cold load. Performance is compared to the original FlashRT
entry points and relevant eager/compiled references without counting Python
packing loops as a kernel speedup baseline.

## SM110 portable SIMT fallback

`grouped_nvfp4_gemm_bf16` also ships a pure-SIMT reference
(`portable_moe_simt.cu`) compiled for `sm_110a`. The correctness test runs the
full case matrix a second time with `FLASHRT_FORCE_SIMT=1`, validating the SIMT
path against the same FP32 reference on any device. Native SM120 tile behavior
is unchanged. Runtime validation of the SM110 artifact on Thor hardware is
pending and is not recorded as passed here.
