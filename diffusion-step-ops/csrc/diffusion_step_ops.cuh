// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace diffusion_step_ops {

void add_bf16_out(const void* a, const void* b, void* out, int64_t n, cudaStream_t stream);

void euler_step_bf16_out(
    const void* latent,
    const void* velocity,
    void* out,
    float dt,
    int64_t n,
    cudaStream_t stream);

void cfg_combine_into_residual_bf16(
    void* residual,
    const void* v_cond,
    const void* v_uncond,
    float beta,
    int64_t n,
    cudaStream_t stream);

void cfg_combine_into_residual_fp16(
    void* residual,
    const void* v_cond,
    const void* v_uncond,
    float beta,
    int64_t n,
    cudaStream_t stream);

void teacher_force_first_frame_bf16(
    void* video_latent,
    const void* cond_latent,
    int b,
    int c,
    int t,
    int h,
    int w,
    cudaStream_t stream);

void motus_decode_postprocess_bf16_to_fp32(
    const void* decoded,
    void* out,
    int b,
    int c,
    int t_in,
    int h,
    int w,
    cudaStream_t stream);

void cast_bf16_to_fp32(const void* src, void* dst, int64_t n, cudaStream_t stream);

void pack_tail_bf16(
    const void* tail,
    void* out,
    int64_t flat_dim,
    int64_t tail_numel,
    cudaStream_t stream);

void add_bias_zero_tail_bf16(
    const void* input,
    const void* bias,
    void* out,
    int64_t rows,
    int64_t cols,
    int64_t valid_cols,
    cudaStream_t stream);

void extract_tail_f32_to_bf16(
    const void* flat,
    void* out,
    int64_t flat_dim,
    int64_t tail_numel,
    cudaStream_t stream);

void add_bias_pair_bf16(
    const void* input,
    const void* bias_a,
    const void* bias_b,
    void* out,
    int64_t rows,
    int64_t hidden,
    cudaStream_t stream);

void unipc_step_f32_bf16(
    const void* sample,
    const void* velocity,
    const void* prev_m1,
    const void* prev_m2,
    const void* prev_last_sample,
    void* next_sample,
    void* current_m,
    void* current_last_sample,
    int64_t n,
    float sigma,
    int corrector_order,
    int predictor_order,
    float c_sample,
    float c_last,
    float c_prev_m1,
    float c_prev_m2,
    float c_curr_m,
    float p_sample,
    float p_curr_m,
    float p_prev_m1,
    cudaStream_t stream);

}  // namespace diffusion_step_ops
}  // namespace flash_rt
