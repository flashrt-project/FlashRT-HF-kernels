#pragma once

#include <torch/all.h>

void nvfp4_sf_linear_to_swizzled(
    torch::Tensor const& scales,
    torch::Tensor& out,
    int64_t D,
    bool is_sfb);
