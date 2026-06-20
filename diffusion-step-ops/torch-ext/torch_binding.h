// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/all.h>

void add_bf16_out(torch::Tensor const& a, torch::Tensor const& b, torch::Tensor& out);
void euler_step_bf16_out(torch::Tensor const& latent, torch::Tensor const& velocity, double dt, torch::Tensor& out);
void cfg_combine_into_residual_bf16(torch::Tensor& residual, torch::Tensor const& v_cond, torch::Tensor const& v_uncond, double beta);
void cfg_combine_into_residual_fp16(torch::Tensor& residual, torch::Tensor const& v_cond, torch::Tensor const& v_uncond, double beta);
void teacher_force_first_frame_bf16(torch::Tensor& video_latent, torch::Tensor const& cond_latent);
void motus_decode_postprocess_bf16_to_fp32(torch::Tensor const& decoded, torch::Tensor& out);
void cast_bf16_to_fp32(torch::Tensor const& src, torch::Tensor& dst);
