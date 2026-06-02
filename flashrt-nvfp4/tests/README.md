# Tests

Implemented test groups:

- `nvfp4_sf_swizzled_bytes` matches the reference allocation formula for common
  and boundary shapes.
- `nvfp4_sf_linear_to_swizzled` matches a CPU/PyTorch reference layout transform.

Planned test groups:

- Fused GEMM epilogues match FlashRT internal outputs and a fake-quant PyTorch
  reference within NVFP4 tolerances.
- Rejected shape and alignment cases.
