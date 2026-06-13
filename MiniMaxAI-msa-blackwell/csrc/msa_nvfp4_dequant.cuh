// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace flashrt_minimax_msa {

void nvfp4_dequant_swizzled_to_bf16_cuda(const uint8_t* packed,
                                         const uint8_t* scale_128x4,
                                         void* out_bf16,
                                         int rows,
                                         int cols,
                                         float global_scale,
                                         void* stream);

}  // namespace flashrt_minimax_msa
