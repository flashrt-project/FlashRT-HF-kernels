// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flashrt_hub {
namespace int8_transformer {

void quantize_int8_static_bf16(const __nv_bfloat16* input, int8_t* output,
                               const float* scale, int n,
                               cudaStream_t stream);

void quantize_int8_rowwise_bf16(const __nv_bfloat16* input, int8_t* output,
                                float* scales, int rows, int cols,
                                cudaStream_t stream);

void quantize_int8_rowwise_static_bf16(const __nv_bfloat16* input, int8_t* output,
                                       const float* scales, int rows, int cols,
                                       cudaStream_t stream);

void dequant_int32_to_bf16(const int32_t* input, __nv_bfloat16* output,
                           const float* act_scale, const float* weight_scale,
                           int n, cudaStream_t stream);

void rms_norm_quantize_int8_rowwise_bf16(const __nv_bfloat16* x,
                                         const __nv_bfloat16* weight,
                                         int8_t* out, float* scales,
                                         int rows, int cols, float eps,
                                         cudaStream_t stream);

void residual_add_rms_norm_quantize_int8_rowwise_bf16(
    __nv_bfloat16* residual, const __nv_bfloat16* x,
    const __nv_bfloat16* weight, int8_t* out, float* scales,
    int rows, int cols, float eps, cudaStream_t stream);

}  // namespace int8_transformer
}  // namespace flashrt_hub

extern "C" int cutlass_int8_rowwise_bf16out(
    void const* A, void const* B,
    void const* act_scale, void const* weight_scale,
    void* D, int M, int N, int K, cudaStream_t stream);

extern "C" int cutlass_int8_rowwise_bf16out_t64x128(
    void const* A, void const* B,
    void const* act_scale, void const* weight_scale,
    void* D, int M, int N, int K, cudaStream_t stream);

extern "C" int cutlass_int8_silu_gated_bf16out(
    void const* act_i8, void const* up_w_i8,
    void const* act_scale, void const* wt_scale,
    void const* gate_buf, void* D,
    int M, int N, int K, cudaStream_t stream);
