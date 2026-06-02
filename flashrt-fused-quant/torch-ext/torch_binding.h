#pragma once

#include <torch/all.h>

void silu_mul_quant_nvfp4_swizzled_bf16(
    torch::Tensor const& gate,
    torch::Tensor const& up,
    torch::Tensor& packed,
    torch::Tensor& scales);

void silu_mul_merged_quant_nvfp4_swizzled_bf16(
    torch::Tensor const& merged_gate_up,
    torch::Tensor& packed,
    torch::Tensor& scales);
