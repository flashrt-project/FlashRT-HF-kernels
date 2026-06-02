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

void qkv_split_norm_rope_bf16(
    const void* packed_qkv,
    const void* norm_q_weight,
    const void* norm_k_weight,
    const void* freqs_re,
    const void* freqs_im,
    void* q_out,
    void* k_out,
    int batch,
    int tokens,
    int heads,
    int head_dim,
    int seq_len,
    float eps,
    cudaStream_t stream);

}  // namespace vla_video
}  // namespace flash_rt
