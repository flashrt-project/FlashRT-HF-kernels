// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/all.h>

void fp8_conv3d_v18_ncdhw_res_bf16out(
    torch::Tensor const& cache_x,
    torch::Tensor const& new_x,
    torch::Tensor const& weight,
    torch::Tensor const& bias,
    torch::Tensor const& residual,
    double alpha,
    torch::Tensor& out);
