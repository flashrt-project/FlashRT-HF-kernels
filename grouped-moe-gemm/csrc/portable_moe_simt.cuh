// SPDX-License-Identifier: Apache-2.0
//
// Portable SIMT reference grouped NVFP4 GEMM (see portable_moe_simt.cu).

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace gemm {

int moe_gemm_bf16_simt(
    const void* A_tiled, const void* B_stack, const void* SFA_tiled,
    const void* SFB_stack, void* D, const void* alpha_stack,
    const void* tile_expert, int num_tiles, int tile_rows, int N, int K,
    long input_scale_stride, long w_stride, long sfb_stride,
    cudaStream_t stream);

}  // namespace gemm
}  // namespace flash_rt
