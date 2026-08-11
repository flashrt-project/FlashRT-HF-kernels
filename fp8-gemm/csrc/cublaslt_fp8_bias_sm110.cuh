#pragma once

#include <cuda_runtime.h>

enum class FlashRtFp8BiasEpilogue : int {
  kNone = -1,
  kBias = 0,
  kBiasGelu = 1,
};

int fp8_linear_cublaslt_bf16(
    const void* input_fp8,
    const void* weight_fp8,
    const void* bias_bf16,
    void* out_bf16,
    int M,
    int N,
    int K,
    float alpha,
    float beta,
    FlashRtFp8BiasEpilogue epilogue,
    cudaStream_t stream);

int fp8_linear_bias_sm110_bf16(
    const void* input_fp8,
    const void* weight_fp8,
    const void* bias_bf16,
    void* out_bf16,
    int M,
    int N,
    int K,
    float alpha,
    float beta,
    FlashRtFp8BiasEpilogue epilogue,
    cudaStream_t stream);
