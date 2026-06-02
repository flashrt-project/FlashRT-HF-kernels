#pragma once

#include <torch/all.h>

void nvfp4_w4a4_decode_matvec_bf16out(
    torch::Tensor const& a_packed,
    torch::Tensor const& b_packed,
    torch::Tensor const& sfa,
    torch::Tensor const& sfb,
    torch::Tensor& out,
    double alpha);
