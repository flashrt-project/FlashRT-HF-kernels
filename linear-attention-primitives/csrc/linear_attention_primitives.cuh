// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt::linear_attention_primitives {

void bf16_matvec(
    const __nv_bfloat16* x,
    const __nv_bfloat16* w,
    __nv_bfloat16* out,
    int n,
    int k,
    cudaStream_t stream);

void bf16_smallm_matmul(
    const __nv_bfloat16* x,
    const __nv_bfloat16* w,
    __nv_bfloat16* out,
    int m,
    int n,
    int k,
    cudaStream_t stream);

void split_qkv_gqa_bf16(
    const __nv_bfloat16* packed,
    __nv_bfloat16* q,
    __nv_bfloat16* k,
    __nv_bfloat16* v,
    int rows,
    int q_heads,
    int kv_heads,
    int head_dim,
    cudaStream_t stream);

void split_qkv_broadcast_bf16(
    const __nv_bfloat16* packed,
    __nv_bfloat16* q,
    __nv_bfloat16* k,
    __nv_bfloat16* v,
    int rows,
    int q_heads,
    int kv_heads,
    int v_heads,
    int head_dim,
    cudaStream_t stream);

void partial_rope_qk_bf16(
    const __nv_bfloat16* q_in,
    const __nv_bfloat16* k_in,
    const __nv_bfloat16* cos,
    const __nv_bfloat16* sin,
    __nv_bfloat16* q_out,
    __nv_bfloat16* k_out,
    int rows,
    int q_heads,
    int k_heads,
    int head_dim,
    int rope_dim,
    cudaStream_t stream);

void gated_delta_prepare_bf16(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    const float* neg_exp_a_log,
    const float* dt_bias,
    __nv_bfloat16* g_out,
    __nv_bfloat16* beta_out,
    int rows,
    int heads,
    int a_stride,
    int b_stride,
    cudaStream_t stream);

}  // namespace flash_rt::linear_attention_primitives
