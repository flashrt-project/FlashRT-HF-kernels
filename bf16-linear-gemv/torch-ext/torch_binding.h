// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/all.h>

void bf16_decode_gemv_bf16(
    torch::Tensor const& x,
    torch::Tensor const& weight,
    double alpha,
    int64_t variant,
    torch::Tensor& out);

void bf16_decode_gemv_unrolled_bf16(
    torch::Tensor const& x,
    torch::Tensor const& weight,
    torch::Tensor& out);
