// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// q/k/v: (B, H, 128) bf16. g/beta: (B, H) bf16. state_in/out:
// (B, H, 128, 128) bf16 (may alias). out: (B, H, 128) bf16.
int gdn_recurrent_inout_stream_bf16(
    const void* q, const void* k, const void* v, const void* g,
    const void* beta, const void* state_in, void* state_out, void* out,
    int B, int num_v_heads, int head_dim, bool use_qk_l2norm,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
