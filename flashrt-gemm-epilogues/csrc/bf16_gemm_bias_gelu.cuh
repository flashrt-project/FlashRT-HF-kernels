#pragma once

#include <cuda_runtime.h>

namespace flash_rt::gemm {

void bf16_gemm_bias(
    const void* A,
    const void* B,
    const void* bias,
    void* D,
    int M,
    int N,
    int K,
    cudaStream_t stream);

void bf16_gemm_bias_gelu(
    const void* A,
    const void* B,
    const void* bias,
    void* D,
    int M,
    int N,
    int K,
    cudaStream_t stream);

}  // namespace flash_rt::gemm
