// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/all.h>

void causal_conv1d_bf16(torch::Tensor const& x, torch::Tensor const& w,
                        torch::Tensor const& bias, torch::Tensor& out,
                        bool has_bias, bool apply_silu);
void causal_conv1d_update_bf16(torch::Tensor const& x_new, torch::Tensor const& w,
                               torch::Tensor const& bias, torch::Tensor& state,
                               torch::Tensor& out, bool has_bias, bool apply_silu);
void causal_conv1d_update_inout_bf16(torch::Tensor const& x_new, torch::Tensor const& w,
                                     torch::Tensor const& bias,
                                     torch::Tensor const& state_in,
                                     torch::Tensor& state_out,
                                     torch::Tensor& out,
                                     bool has_bias, bool apply_silu);
void causal_conv1d_update_chunk_bf16(torch::Tensor const& x, torch::Tensor const& w,
                                     torch::Tensor const& bias, torch::Tensor& state,
                                     torch::Tensor& out, bool has_bias, bool apply_silu);
void causal_conv1d_update_chunk_parallel_bf16(torch::Tensor const& x, torch::Tensor const& w,
                                              torch::Tensor const& bias, torch::Tensor& state,
                                              torch::Tensor& out, bool has_bias, bool apply_silu);
void causal_conv1d_update_chunk_parallel_gqa_bf16(torch::Tensor const& x, torch::Tensor const& w,
                                                  torch::Tensor const& bias, torch::Tensor& state,
                                                  torch::Tensor& q16, torch::Tensor& k16,
                                                  torch::Tensor& v48, bool has_bias,
                                                  bool apply_silu);
