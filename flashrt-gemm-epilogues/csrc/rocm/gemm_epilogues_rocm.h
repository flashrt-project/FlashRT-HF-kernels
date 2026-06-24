#pragma once

#include <hip/hip_runtime.h>

namespace flash_rt {
namespace gemm {

void bf16_gemm(
    const void* a,
    const void* b,
    void* out,
    int M,
    int N,
    int K,
    hipStream_t stream);

void bf16_gemm_bias(
    const void* a,
    const void* b,
    const void* bias,
    void* out,
    int M,
    int N,
    int K,
    hipStream_t stream);

void bf16_gemm_bias_gelu(
    const void* a,
    const void* b,
    const void* bias,
    void* out,
    int M,
    int N,
    int K,
    hipStream_t stream);

}  // namespace gemm

namespace quantize {

void bias_gelu_quantize_fp8_static_bf16(
    const void* in_bf16,
    const void* bias_bf16,
    void* out_fp8,
    const float* act_scale,
    long long M,
    int N,
    hipStream_t stream);

void channel_scale_quantize_fp8_static_bf16(
    const void* in_bf16,
    const void* channel_scale_bf16,
    void* out_fp8,
    const float* act_scale,
    long long M,
    int K,
    hipStream_t stream);

}  // namespace quantize
}  // namespace flash_rt
