// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/all.h>

void ada_rms_norm_style_bf16(
    torch::Tensor const& x,
    torch::Tensor const& weight,
    torch::Tensor const& style,
    double eps,
    torch::Tensor& out,
    torch::Tensor& gate_out);

void gate_residual_ada_norm_fp8_static_bf16(
    torch::Tensor& residual,
    torch::Tensor const& x,
    torch::Tensor const& gate,
    torch::Tensor const& weight,
    torch::Tensor const& style,
    torch::Tensor const& scale,
    double eps,
    torch::Tensor& out,
    torch::Tensor& gate_out);
