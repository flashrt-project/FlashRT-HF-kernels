// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/all.h>

void cholesky_small_fp32_out(
    torch::Tensor const& input,
    torch::Tensor& output);
