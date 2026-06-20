// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/all.h>

void bf16_matvec(torch::Tensor const& x, torch::Tensor const& w, torch::Tensor& out);
void bf16_smallm_matmul(torch::Tensor const& x, torch::Tensor const& w, torch::Tensor& out);
void split_qkv_broadcast_bf16(torch::Tensor const& packed, torch::Tensor& q, torch::Tensor& k, torch::Tensor& v,
                              int64_t q_heads, int64_t kv_heads, int64_t v_heads, int64_t head_dim);
void partial_rope_qk_bf16(torch::Tensor const& q_in, torch::Tensor const& k_in,
                          torch::Tensor const& cos, torch::Tensor const& sin,
                          torch::Tensor& q_out, torch::Tensor& k_out,
                          int64_t rope_dim);
void gated_delta_prepare_bf16(torch::Tensor const& a, torch::Tensor const& b,
                              torch::Tensor const& neg_exp_a_log, torch::Tensor const& dt_bias,
                              torch::Tensor& g_out, torch::Tensor& beta_out,
                              int64_t a_stride, int64_t b_stride);
