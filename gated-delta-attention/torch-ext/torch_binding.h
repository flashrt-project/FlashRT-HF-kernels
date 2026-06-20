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
void lin_split_qkv_broadcast_bf16(torch::Tensor const& conv_out,
                                  torch::Tensor& q48,
                                  torch::Tensor& k48,
                                  torch::Tensor& v48);
void lin_split_qkv_gqa_bf16(torch::Tensor const& conv_out,
                            torch::Tensor& q16,
                            torch::Tensor& k16,
                            torch::Tensor& v48);
void split_q_gate_bf16(torch::Tensor const& q_proj,
                       torch::Tensor& q_pre,
                       torch::Tensor& gate);
void gdn_gating_bf16(torch::Tensor const& a, torch::Tensor const& b,
                     torch::Tensor const& neg_exp_A_log,
                     torch::Tensor const& dt_bias,
                     torch::Tensor& g_out,
                     torch::Tensor& beta_out);
void gdn_gating_strided_bf16(torch::Tensor const& a, torch::Tensor const& b,
                             torch::Tensor const& neg_exp_A_log,
                             torch::Tensor const& dt_bias,
                             torch::Tensor& g_out,
                             torch::Tensor& beta_out,
                             int64_t a_stride,
                             int64_t b_stride);
void gdn_chunk_from_conv_smem_bf16(torch::Tensor const& conv_out,
                                   torch::Tensor const& a,
                                   torch::Tensor const& b,
                                   torch::Tensor const& neg_exp_A_log,
                                   torch::Tensor const& dt_bias,
                                   torch::Tensor& state,
                                   torch::Tensor& out,
                                   bool use_qk_l2norm);
void gdn_wy_norm_cumsum_pack_qk_bf16(torch::Tensor const& q16,
                                     torch::Tensor const& k16,
                                     torch::Tensor const& g,
                                     torch::Tensor& q16_l2,
                                     torch::Tensor& k16_l2,
                                     torch::Tensor& q_pack_hv,
                                     torch::Tensor& k_pack_hk,
                                     torch::Tensor& g_cumsum);
void gdn_wy_kkt_b64_bf16(torch::Tensor const& k16_l2,
                         torch::Tensor const& beta,
                         torch::Tensor const& g_cumsum,
                         torch::Tensor& A);
void gdn_wy_solve_tril_b64_f32(torch::Tensor const& A,
                               torch::Tensor& Ai,
                               int64_t S);
void gdn_wy_recompute_wu_b64_bf16(torch::Tensor const& k16_l2,
                                  torch::Tensor const& v48,
                                  torch::Tensor const& beta,
                                  torch::Tensor const& g_cumsum,
                                  torch::Tensor const& Ai,
                                  torch::Tensor& w48,
                                  torch::Tensor& u48);
void gdn_wy_chunk_h_b64_bf16(torch::Tensor const& k16_l2,
                             torch::Tensor const& u48,
                             torch::Tensor const& w48,
                             torch::Tensor const& g_cumsum,
                             torch::Tensor& state,
                             torch::Tensor& h0,
                             torch::Tensor& v_new);
void gdn_wy_output_o_b64_bf16(torch::Tensor const& q16_l2,
                              torch::Tensor const& k16_l2,
                              torch::Tensor const& v_new,
                              torch::Tensor const& h0,
                              torch::Tensor const& g_cumsum,
                              torch::Tensor& out);
