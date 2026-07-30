// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

void layer_norm_to_fp8_block128_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* gamma,
    const __nv_bfloat16* beta, __nv_fp8_e4m3* out, float* scale,
    int rows, int dim, float eps, cudaStream_t stream);

void gelu_tanh_to_fp8_block128_bf16(
    const __nv_bfloat16* x, __nv_fp8_e4m3* out, float* scale,
    int rows, int dim, cudaStream_t stream);

void gelu_tanh_bias_to_fp8_block128_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* bias,
    __nv_fp8_e4m3* out, float* scale, int rows, int dim,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
