// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

namespace flash_rt::fused_fp4 {

void adaptive_rms_norm_fp8_static_fp16(
    const void* x, const void* style, void* out, void* gate,
    int rows, int dim, const float* scale, cudaStream_t stream);

void gated_residual_adaptive_rms_norm_fp8_static_fp16(
    const void* x, const void* previous_gate, void* residual,
    const void* style, void* out, void* gate, int rows, int dim,
    const float* scale, cudaStream_t stream);

}  // namespace flash_rt::fused_fp4
