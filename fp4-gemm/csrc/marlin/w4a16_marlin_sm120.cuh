#pragma once

#include <cuda_runtime.h>

namespace flash_rt::gemm {

int w4a16_marlin_sm120_bf16(const void* a, const void* b_marlin,
                            const void* scales_marlin,
                            const void* global_scale, void* workspace,
                            void* out, int m, int n, int k, int lda,
                            cudaStream_t stream);

}  // namespace flash_rt::gemm
