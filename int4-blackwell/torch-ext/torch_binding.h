// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/types.h>

torch::Tensor run_codebook_probe(torch::Tensor const& cubin, int64_t device);
void run_mma_probe(torch::Tensor const& cubin, torch::Tensor& output,
                   int64_t iterations, int64_t blocks, int64_t launches,
                   int64_t device);
