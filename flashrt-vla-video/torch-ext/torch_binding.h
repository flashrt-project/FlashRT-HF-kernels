#pragma once

#include <torch/all.h>

void q_norm_rope_bf16(
    torch::Tensor const& q,
    torch::Tensor const& weight,
    torch::Tensor const& cos,
    torch::Tensor const& sin,
    torch::Tensor& out,
    double eps);

void k_norm_rope_v_cache_bf16(
    torch::Tensor const& k,
    torch::Tensor const& v,
    torch::Tensor const& weight,
    torch::Tensor const& cos,
    torch::Tensor const& sin,
    torch::Tensor& k_out,
    torch::Tensor& v_out,
    double eps);
