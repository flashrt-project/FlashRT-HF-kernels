# Torch Extension

Implemented draft Tensor wrappers:

- `nvfp4_w4a4_decode_matvec_bf16out`

Planned Tensor wrappers:

- `nvfp4_w4a4_smallm_warpsplit_bf16out`
- `tiny_fp8_smallm_gemm_bf16out`

Do not expose a generic dispatcher until shape-grid benchmark evidence exists.
