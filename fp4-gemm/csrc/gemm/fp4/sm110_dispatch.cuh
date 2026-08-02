#pragma once

#include <cuda_runtime_api.h>

namespace flash_rt::hub {

using Sm110GemmDispatch = void (*)(
    const void* a,
    const void* b,
    void* out,
    int m,
    int n,
    int k,
    const void* sfa,
    const void* sfb,
    float alpha,
    int variant,
    cudaStream_t stream);

extern Sm110GemmDispatch sm110_gemm_dispatch;

}  // namespace flash_rt::hub
