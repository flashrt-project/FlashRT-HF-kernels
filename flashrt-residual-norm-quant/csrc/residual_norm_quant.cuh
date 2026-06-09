// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace residual_norm_quant {

void rms_norm_bf16(
    const void* x_bf16,
    const void* weight_bf16,
    void* out_bf16,
    int rows,
    int dim,
    float eps,
    cudaStream_t stream);

void layer_norm_bf16(
    const void* x_bf16,
    const void* weight_bf16,
    const void* bias_bf16,
    void* out_bf16,
    int rows,
    int dim,
    float eps,
    cudaStream_t stream);

void rms_norm_quant_fp8_static_bf16(
    const void* x_bf16,
    const void* weight_bf16,
    void* out_fp8,
    int rows,
    int dim,
    float eps,
    const float* scale,
    cudaStream_t stream);

void residual_add_rms_norm_quant_fp8_static_bf16(
    void* residual_bf16,
    const void* x_bf16,
    const void* weight_bf16,
    void* out_fp8,
    int rows,
    int dim,
    float eps,
    const float* scale,
    cudaStream_t stream);

}  // namespace residual_norm_quant
}  // namespace flash_rt
