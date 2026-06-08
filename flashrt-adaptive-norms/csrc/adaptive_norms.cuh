// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace adaptive_norms {

void ada_rms_norm_style_bf16(
    const void* x,
    const void* weight,
    const void* style,
    void* out,
    void* gate_out,
    int rows,
    int dim,
    float eps,
    cudaStream_t stream);

void gate_residual_ada_norm_fp8_static_bf16(
    void* residual,
    const void* x,
    const void* gate,
    const void* weight,
    const void* style,
    void* out,
    void* gate_out,
    int rows,
    int dim,
    float eps,
    const float* scale,
    cudaStream_t stream);

}  // namespace adaptive_norms
}  // namespace flash_rt
