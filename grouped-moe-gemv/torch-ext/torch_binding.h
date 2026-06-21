// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/all.h>

void w4a16_decode_gemv_bf16(torch::Tensor const& x_bf16,
                            torch::Tensor const& weight_packed,
                            torch::Tensor const& sfb,
                            double alpha,
                            torch::Tensor& out);

void grouped_w4a16_gemv_bf16(torch::Tensor const& activations,
                             torch::Tensor const& weight_stack,
                             torch::Tensor const& sfb_stack,
                             torch::Tensor const& alpha_stack,
                             torch::Tensor const& expert_idx,
                             int64_t w_stride,
                             int64_t sfb_stride,
                             torch::Tensor& out);
