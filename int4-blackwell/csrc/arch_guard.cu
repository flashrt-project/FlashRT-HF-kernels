// SPDX-License-Identifier: Apache-2.0

// Executable kernels are reproducible CUDA 13.0 cubins shipped as package
// data. This unit makes the SM12x package architecture contract explicit to
// the builder. SM100/SM103/SM110 use the separately compiled tcgen05 target.
extern "C" __global__ void flashrt_int4_blackwell_arch_guard() {}
