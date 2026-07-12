// SPDX-License-Identifier: Apache-2.0

// Executable kernels are reproducible CUDA 13.0 cubins shipped as package
// data. This unit makes the package architecture contract explicit to the
// builder and prevents generation of non-SM120 variants.
extern "C" __global__ void flashrt_int4_blackwell_arch_guard() {}
