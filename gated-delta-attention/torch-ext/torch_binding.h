// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/all.h>

void gated_delta_recurrent_bf16(torch::Tensor const& q, torch::Tensor const& k,
                                torch::Tensor const& v, torch::Tensor const& g,
                                torch::Tensor const& beta, torch::Tensor& state,
                                torch::Tensor& out, bool use_qk_l2norm);
void gated_delta_recurrent_inout_bf16(torch::Tensor const& q, torch::Tensor const& k,
                                      torch::Tensor const& v, torch::Tensor const& g,
                                      torch::Tensor const& beta,
                                      torch::Tensor const& state_in,
                                      torch::Tensor& state_out,
                                      torch::Tensor& out,
                                      bool use_qk_l2norm);
void gated_delta_recurrent_f32state_bf16io(torch::Tensor const& q, torch::Tensor const& k,
                                           torch::Tensor const& v, torch::Tensor const& g,
                                           torch::Tensor const& beta,
                                           torch::Tensor& state_f32,
                                           torch::Tensor& out,
                                           bool use_qk_l2norm);
void gated_delta_chunk_bf16(torch::Tensor const& q, torch::Tensor const& k,
                            torch::Tensor const& v, torch::Tensor const& g,
                            torch::Tensor const& beta, torch::Tensor& state,
                            torch::Tensor& out, bool use_qk_l2norm);
void gated_delta_chunk_smem_bf16(torch::Tensor const& q, torch::Tensor const& k,
                                 torch::Tensor const& v, torch::Tensor const& g,
                                 torch::Tensor const& beta, torch::Tensor& state,
                                 torch::Tensor& out, bool use_qk_l2norm);
