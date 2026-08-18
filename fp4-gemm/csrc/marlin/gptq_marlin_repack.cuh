#pragma once

#include <cuda_runtime.h>

namespace flash_rt::gemm {

int repack_nvfp4_to_marlin_sm120(const void* qweight_kn, void* out,
                                 int size_k, int size_n,
                                 cudaStream_t stream);

}  // namespace flash_rt::gemm
