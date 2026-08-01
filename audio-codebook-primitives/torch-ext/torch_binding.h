#pragma once

#include <torch/all.h>

void delayed_codebook_argmax_embed_bf16(
    torch::Tensor const& logits, torch::Tensor const& codebook,
    int64_t delay, int64_t boc, torch::Tensor& codes,
    torch::Tensor& embedding);

void delayed_codebook_sample_embed_bf16(
    torch::Tensor const& logits, torch::Tensor const& codebook,
    int64_t delay, int64_t boc, double temperature,
    int64_t seed, int64_t step, torch::Tensor& codes,
    torch::Tensor& embedding);
