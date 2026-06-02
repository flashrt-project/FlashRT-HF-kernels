#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace vla_video {

void q_norm_rope_bf16(
    const void* q,
    const void* weight,
    const void* cos,
    const void* sin,
    void* out,
    int rows,
    float eps,
    cudaStream_t stream);

void k_norm_rope_v_cache_bf16(
    const void* k,
    const void* v,
    const void* weight,
    const void* cos,
    const void* sin,
    void* k_out,
    void* v_out,
    int rows,
    float eps,
    cudaStream_t stream);

}  // namespace vla_video
}  // namespace flash_rt
