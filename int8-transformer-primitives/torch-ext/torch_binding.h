// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/all.h>

void quantize_int8_static_bf16(torch::Tensor const& input,
                               torch::Tensor const& scale,
                               torch::Tensor& out);
void quantize_int8_rowwise_bf16(torch::Tensor const& input,
                                torch::Tensor& out,
                                torch::Tensor& scales);
void quantize_int8_rowwise_static_bf16(torch::Tensor const& input,
                                       torch::Tensor const& scales,
                                       torch::Tensor& out);
void rms_norm_quantize_int8_rowwise_bf16(torch::Tensor const& x,
                                         torch::Tensor const& weight,
                                         double eps,
                                         torch::Tensor& out,
                                         torch::Tensor& scales);
void residual_add_rms_norm_quantize_int8_rowwise_bf16(torch::Tensor& residual,
                                                      torch::Tensor const& x,
                                                      torch::Tensor const& weight,
                                                      double eps,
                                                      torch::Tensor& out,
                                                      torch::Tensor& scales);
void int8_rowwise_linear_bf16(torch::Tensor const& input_i8,
                              torch::Tensor const& weight_i8,
                              torch::Tensor const& input_scale,
                              torch::Tensor const& weight_scale,
                              torch::Tensor& out,
                              int64_t variant);
void int8_silu_gated_linear_bf16(torch::Tensor const& input_i8,
                                 torch::Tensor const& up_weight_i8,
                                 torch::Tensor const& input_scale,
                                 torch::Tensor const& weight_scale,
                                 torch::Tensor const& gate,
                                 torch::Tensor& out);
