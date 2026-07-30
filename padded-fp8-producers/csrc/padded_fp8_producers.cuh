// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace flash_rt::padded_fp8 {

void adaptive_rms_norm_quant_fp8_padded_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    const __nv_bfloat16* gamma, const __nv_bfloat16* beta,
    const float* scale, __nv_fp8_e4m3* output, int batch, int rows,
    int padded_rows, int dim, float eps, cudaStream_t stream);

void residual_add_adaptive_rms_norm_quant_fp8_padded_bf16(
    const __nv_bfloat16* residual, const __nv_bfloat16* input,
    const __nv_bfloat16* weight, const __nv_bfloat16* gamma,
    const __nv_bfloat16* beta, const float* scale,
    __nv_bfloat16* residual_out, __nv_fp8_e4m3* output, int batch, int rows,
    int padded_rows, int dim, float eps, cudaStream_t stream);

void swiglu_quant_fp8_padded_bf16(
    const __nv_bfloat16* gate, const __nv_bfloat16* up, const float* scale,
    __nv_fp8_e4m3* output, int rows, int padded_rows, int dim,
    cudaStream_t stream);

void swiglu_merged_quant_fp8_padded_bf16(
    const __nv_bfloat16* gate_up, const float* scale, __nv_fp8_e4m3* output,
    int rows, int padded_rows, int dim, cudaStream_t stream);

void swiglu_merged_quant_fp8_padded_fp16(
    const __half* gate_up, const float* scale, __nv_fp8_e4m3* output,
    int rows, int padded_rows, int dim, cudaStream_t stream);

}  // namespace flash_rt::padded_fp8
