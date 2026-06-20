// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/all.h>

int64_t sfa_size_bytes(int64_t rows, int64_t dim, bool is_sfb);
int64_t sfa_size_bytes_for(torch::Tensor const& anchor, int64_t rows, int64_t dim, bool is_sfb);
