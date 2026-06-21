// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/all.h>

void gated_delta_recurrent_seq_bf16(torch::Tensor const& q,
                                    torch::Tensor const& k,
                                    torch::Tensor const& v,
                                    torch::Tensor const& g,
                                    torch::Tensor const& beta,
                                    torch::Tensor& state,
                                    torch::Tensor& out,
                                    bool use_qk_l2norm);
