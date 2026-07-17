# Source Sync

The initial W4 decode implementation was derived from the production FlashRT
W4A16 SM120 kernels:

- `csrc/kernels/w4a16_gemm_sm120.cu`
- `csrc/kernels/w4a16_matvec_sm120.cu`

The Hub package adds generic Tensor bindings, multi-row weight reuse, W8
weight-only kernels, FFN region wrappers, strict shape rejection, fake-op
registration, tests, and benchmarks. Upstream files are read-only inputs; Hub
changes are maintained in this package.
