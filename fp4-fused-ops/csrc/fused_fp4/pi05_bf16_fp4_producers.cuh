// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt::fused_fp4 {

int gelu_mul_nvfp4_bf16(
    const __nv_bfloat16* merged, const __nv_bfloat16* inv_s,
    uint8_t* packed, uint8_t* sfa, int rows, int hidden,
    cudaStream_t stream);

int rms_norm_mul_nvfp4_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* inv_s,
    uint8_t* packed, uint8_t* sfa, int rows, int dim, float eps,
    cudaStream_t stream);

int residual_add_rms_norm_nvfp4_bf16(
    __nv_bfloat16* residual, const __nv_bfloat16* x,
    const __nv_bfloat16* inv_s, uint8_t* packed, uint8_t* sfa,
    int rows, int dim, float eps, cudaStream_t stream);

int layer_norm_fp8_vec_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* gamma,
    const __nv_bfloat16* beta, void* out_fp8,
    int rows, int dim, float eps, cudaStream_t stream);

int layer_norm_mul_nvfp4_vec_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* gamma,
    const __nv_bfloat16* beta, const __nv_bfloat16* inv_s,
    uint8_t* packed, uint8_t* sfa,
    int rows, int dim, float eps, cudaStream_t stream);

}  // namespace flash_rt::fused_fp4
