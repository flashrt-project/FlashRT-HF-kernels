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

}  // namespace diffusion_step_ops
}  // namespace flash_rt
