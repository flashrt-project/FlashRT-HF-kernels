// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/all.h>

void fill_neginf_bf16(torch::Tensor& dst);
void add_bias_bf16_(torch::Tensor& data, torch::Tensor const& bias);
void repeat_interleave_heads_bf16(torch::Tensor const& src, int64_t repeat, torch::Tensor& dst);
void text_gather_bf16(torch::Tensor const& src, int64_t batch, int64_t seq, torch::Tensor& dst);
void text_scatter_bf16(torch::Tensor& dst, torch::Tensor const& src, int64_t batch, int64_t seq);
void rope_rotate_half_bf16_(torch::Tensor& x, torch::Tensor const& cos, torch::Tensor const& sin);
void qk_rmsnorm_rope_bf16_(torch::Tensor& qk, torch::Tensor const& weight,
                           torch::Tensor const& cos, torch::Tensor const& sin,
                           double eps);
