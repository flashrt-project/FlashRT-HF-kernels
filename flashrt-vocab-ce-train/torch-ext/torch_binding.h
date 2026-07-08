// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/all.h>

torch::Tensor flashrt_training_package_marker(torch::Tensor x);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
vocab_ce_fwd_stream(const torch::Tensor& hidden, const torch::Tensor& weight,
                    const torch::Tensor& labels);
