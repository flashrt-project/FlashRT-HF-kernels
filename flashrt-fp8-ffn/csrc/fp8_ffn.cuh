// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace fp8_ffn {

void fp8_gemm_descale_bf16out(
    const void* input_fp8,
    const void* weight_fp8,
    void* out_bf16,
    int M,
    int N,
    int K,
    const float* input_scale,
    const float* weight_scale,
    cudaStream_t stream);

void bias_gelu_quantize_fp8_static_bf16(
    const void* input_bf16,
    const void* bias_bf16,
    void* out_fp8,
    const float* scale,
    long long M,
    int N,
    cudaStream_t stream);

void add_bias_bf16(
    void* input_bf16,
    const void* bias_bf16,
    long long M,
    int N,
    cudaStream_t stream);

}  // namespace fp8_ffn
}  // namespace flash_rt
