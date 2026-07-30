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
void qk_pair_rmsnorm_rope_bf16(
    torch::Tensor const& q, torch::Tensor const& k,
    torch::Tensor const& q_weight, torch::Tensor const& k_weight,
    torch::Tensor const& cos, torch::Tensor const& sin,
    double eps, torch::Tensor& q_out, torch::Tensor& k_out);
void gather_rows_bf16(torch::Tensor const& src, torch::Tensor const& row_indices, torch::Tensor& dst);
void scatter_rows_bf16(torch::Tensor const& src, torch::Tensor const& row_indices, torch::Tensor& dst);
