#pragma once
#include <torch/types.h>

void grouped_nvfp4_gemm_bf16_out(torch::Tensor const &, torch::Tensor const &,
                                 torch::Tensor const &, torch::Tensor const &,
                                 torch::Tensor const &, torch::Tensor const &,
                                 int64_t, int64_t, int64_t, int64_t,
                                 torch::Tensor &);
