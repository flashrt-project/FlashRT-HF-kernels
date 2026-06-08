// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/all.h>

void rms_norm_bf16(
    torch::Tensor const& x,
    torch::Tensor const& weight,
    double eps,
    torch::Tensor& out);

void rms_norm_quant_fp8_static_bf16(
    torch::Tensor const& x,
    torch::Tensor const& weight,
    torch::Tensor const& scale,
    double eps,
    torch::Tensor& out);

void residual_add_rms_norm_quant_fp8_static_bf16(
    torch::Tensor& residual,
    torch::Tensor const& x,
    torch::Tensor const& weight,
    torch::Tensor const& scale,
    double eps,
    torch::Tensor& out);
