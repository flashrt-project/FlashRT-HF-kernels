# CUDA source

`cholesky_small_fp32.cu` contains the standalone CUDA implementation.
`cholesky_small_fp32.cuh` exposes only a pointer launcher used by the package's
Tensor binding; raw pointers are not part of the public Python API.
