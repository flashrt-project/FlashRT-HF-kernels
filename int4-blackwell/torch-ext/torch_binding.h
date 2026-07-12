// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/types.h>

torch::Tensor run_codebook_probe(torch::Tensor const& cubin, int64_t device);
torch::Tensor run_mma_probe(torch::Tensor const& cubin, int64_t iterations,
                            int64_t blocks, int64_t device);
