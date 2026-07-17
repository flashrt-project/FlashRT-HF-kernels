# tests

Correctness tests for the FP8 GEMM and FFN block wrappers. The BF16-entry
matrix covers the vectorized production shapes, explicit M padding, staged-op
bit-exactness, output dtype, `torch.compile(fullgraph=True)`, and CUDA Graph
replay.
