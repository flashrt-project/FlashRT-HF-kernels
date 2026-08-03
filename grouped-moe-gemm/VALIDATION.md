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
is unchanged.

### SM110 (NVIDIA Thor) real-hardware validation

Validated on an NVIDIA Thor (`sm_110a`) device with PyTorch 2.9.1+cu130 and
CUDA 13.2 against the FP32 dequantize-then-matmul reference:

- tile=16 K=64: rel L2 0.0, max rel 0.0
- tile=16 K=2048: rel L2 1.3e-5, max rel 3.7e-4
- tile=64 K=2048: rel L2 3.6e-5, max rel 1.5e-3

All within the package gate (`rel_l2 <= 2.5e-3`, `max_rel <= 0.01`,
`cosine >= 0.999`).
