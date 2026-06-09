// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace qkv_cache_rope {

void decode_q_norm_rope_stage_bf16(
    const void* q_pre,
    const void* q_norm_w,
    const void* cos,
    const void* sin,
    void* q_buf_dst,
    int n_q_heads,
    float eps,
    cudaStream_t stream);

void decode_k_norm_rope_kvwrite_bf16(
    const void* k_pre,
    const void* v_pre,
    const void* k_norm_w,
    const void* cos,
    const void* sin,
    void* k_cache_dst,
    void* v_cache_dst,
    int n_kv_heads,
    float eps,
    cudaStream_t stream);

void decode_k_norm_rope_kvwrite_devpos_bf16(
    const void* k_pre,
    const void* v_pre,
    const void* k_norm_w,
    const void* cos,
    const void* sin,
    void* k_cache_base,
    void* v_cache_base,
    const void* cur_pos,
    int row_elems,
    int n_kv_heads,
    float eps,
    cudaStream_t stream);

void qkv_split_rope_kvcache_bf16(
    const void* packed_qkv,
    const void* rope,
    void* q_out,
    void* k_cache,
    void* v_cache,
    int batch,
    int seq_len,
    int max_seq_len,
    int q_heads,
    int kv_heads,
    int head_dim,
    int cache_offset,
    cudaStream_t stream);

void qkv_split_bf16(
    const void* packed_qkv,
    void* q_out,
    void* k_out,
    void* v_out,
    int batch,
    int seq_len,
    int heads,
    int head_dim,
    cudaStream_t stream);

void qkv_split_norm_rope_bf16(
    const void* packed_qkv,
    const void* norm_q_w,
    const void* norm_k_w,
    const float* freqs_re,
    const float* freqs_im,
    void* q_rope_out,
    void* k_rope_out,
    int batch,
    int seq_len,
    int heads,
    int head_dim,
    int rope_seq_len,
    float eps,
    cudaStream_t stream);

void qkv_split_bias_norm_rope_v_bf16(
    const void* packed_qkv,
    const void* qkv_bias,
    const void* norm_q_w,
    const void* norm_k_w,
    const float* freqs_re,
    const float* freqs_im,
    void* q_rope_out,
    void* k_rope_out,
    void* v_out,
    int batch,
    int seq_len,
    int heads,
    int head_dim,
    int rope_seq_len,
    float eps,
    cudaStream_t stream);

void qkv_split_bias_norm_rope_v_cat_bf16(
    const void* packed_qkv,
    const void* qkv_bias,
    const void* norm_q_w,
    const void* norm_k_w,
    const float* freqs_re,
    const float* freqs_im,
    void* q_cat_out,
    void* k_cat_out,
    void* v_cat_out,
    int batch,
    int total_seq_len,
    int video_offset,
    int video_seq_len,
    int heads,
    int head_dim,
    int rope_seq_len,
    float eps,
    cudaStream_t stream);

void qkv_split_norm2_cat_bf16(
    const void* packed_a,
    const void* norm_a_q_w,
    const void* norm_a_k_w,
    const void* packed_u,
    const void* norm_u_q_w,
    const void* norm_u_k_w,
    void* q_cat_out,
    void* k_cat_out,
    void* v_cat_out,
    int batch,
    int total_seq_len,
    int video_seq_len,
    int action_seq_len,
    int und_seq_len,
    int heads,
    int head_dim,
    float eps_a,
    float eps_u,
    cudaStream_t stream);

void qkv_split_joint3_cat_bf16(
    const void* packed_v,
    const void* qkv_v_bias,
    const void* norm_v_q_w,
    const void* norm_v_k_w,
    const float* freqs_re,
    const float* freqs_im,
    const void* packed_a,
    const void* norm_a_q_w,
    const void* norm_a_k_w,
    const void* packed_u,
    const void* norm_u_q_w,
    const void* norm_u_k_w,
    void* q_cat_out,
    void* k_cat_out,
    void* v_cat_out,
    int batch,
    int total_seq_len,
    int video_seq_len,
    int action_seq_len,
    int und_seq_len,
    int heads,
    int head_dim,
    int rope_seq_len,
    float eps_v,
    float eps_a,
    float eps_u,
    cudaStream_t stream);

}  // namespace qkv_cache_rope
}  // namespace flash_rt
