// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/all.h>

void fp8_linear_bf16(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    double alpha,
    int64_t variant,
    torch::Tensor& out);

void fp8_linear_residual_bf16(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    double alpha,
    int64_t variant,
    torch::Tensor& residual);
