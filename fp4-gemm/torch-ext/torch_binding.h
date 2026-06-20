// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/all.h>

void fp4_w4a16_linear_bf16(
    torch::Tensor const& a_packed,
    torch::Tensor const& b_packed,
    torch::Tensor const& sfa,
    torch::Tensor const& sfb,
    torch::Tensor& out,
    double alpha,
    int64_t variant);

void quantize_fp4_sfa_fp16(
    torch::Tensor const& x,
    torch::Tensor& packed,
    torch::Tensor& sfa,
    bool is_sfb);

void dequantize_fp4_sfa_fp16(
    torch::Tensor const& packed,
    torch::Tensor const& sfa,
    torch::Tensor& out,
    bool is_sfb);
