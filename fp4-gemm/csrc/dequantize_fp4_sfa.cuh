// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace flash_rt {
namespace fused_fp4 {

void dequantize_fp4_sfa_fp16(
    const uint8_t* packed,
    const uint8_t* sfa,
    __half* out,
    int rows,
    int dim,
    bool is_sfb,
    cudaStream_t stream);

}  // namespace fused_fp4
}  // namespace flash_rt
