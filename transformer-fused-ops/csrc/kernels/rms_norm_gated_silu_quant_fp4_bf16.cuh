// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

int rms_norm_gated_silu_quant_fp4_bf16(
    const void* x, const void* gate, const void* weight, void* out,
    void* packed, void* sfa, int rows, int dim, float eps,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
