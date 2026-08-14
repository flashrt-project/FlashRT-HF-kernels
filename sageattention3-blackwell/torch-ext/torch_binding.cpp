// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#include "registration.h"

void scaled_fp4_quant(torch::Tensor const&, torch::Tensor const&, torch::Tensor const&, int);
void scaled_fp4_quant_permute(torch::Tensor const&, torch::Tensor const&, torch::Tensor const&, int);
void scaled_fp4_quant_trans(torch::Tensor const&, torch::Tensor const&, torch::Tensor const&, int);
void scaled_fp4_quant_centered_q(torch::Tensor const&, torch::Tensor const&,
                                 torch::Tensor const&, torch::Tensor const&);
void scaled_fp4_quant_centered_k(torch::Tensor const&, torch::Tensor const&,
                                 torch::Tensor const&, torch::Tensor const&, torch::Tensor&);

void mha_fwd_static(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, int, torch::Tensor&, torch::Tensor&, torch::Tensor&,
    float, bool, bool, bool);

namespace {

void quantize_q(torch::Tensor const& x, torch::Tensor& packed, torch::Tensor& sf) {
  scaled_fp4_quant(x, packed, sf, 2);
}

void quantize_k(torch::Tensor const& x, torch::Tensor& packed, torch::Tensor& sf) {
  scaled_fp4_quant_permute(x, packed, sf, 2);
}

void quantize_v(torch::Tensor const& x, torch::Tensor& packed, torch::Tensor& sf) {
  scaled_fp4_quant_trans(x, packed, sf, 2);
}

void quantize_q_centered(torch::Tensor const& x, torch::Tensor const& mean,
                         torch::Tensor& packed, torch::Tensor& sf) {
  scaled_fp4_quant_centered_q(x, mean, packed, sf);
}

void quantize_k_centered(torch::Tensor const& x, torch::Tensor const& mean,
                         torch::Tensor& packed, torch::Tensor& sf,
                         torch::Tensor& centered_hnd) {
  scaled_fp4_quant_centered_k(x, mean, packed, sf, centered_hnd);
}

void attention_static(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& sfq, const torch::Tensor& sfk,
    const torch::Tensor& sfv, const torch::Tensor& delta_s,
    int64_t unpadded_k, double softmax_scale, bool causal,
    bool per_block_mean, bool bf16_output, torch::Tensor& out,
    torch::Tensor& softmax_lse, torch::Tensor& semaphore) {
  mha_fwd_static(q, k, v, sfq, sfk, sfv, delta_s,
                 static_cast<int>(unpadded_k), out, softmax_lse, semaphore,
                 static_cast<float>(softmax_scale), causal, per_block_mean,
                 bf16_output);
}

}  // namespace

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("quantize_q_fp4_nhd(Tensor x, Tensor! packed, Tensor! sf) -> ()");
  ops.def("quantize_k_fp4_nhd(Tensor x, Tensor! packed, Tensor! sf) -> ()");
  ops.def("quantize_v_fp4_nhd(Tensor x, Tensor! packed, Tensor! sf) -> ()");
  ops.def("quantize_q_fp4_centered_nhd(Tensor x, Tensor mean, Tensor! packed, Tensor! sf) -> ()");
  ops.def("quantize_k_fp4_centered_nhd(Tensor x, Tensor mean, Tensor! packed, Tensor! sf, Tensor! centered_hnd) -> ()");
  ops.def("blockscaled_fp4_attention_static(Tensor q, Tensor k, Tensor v, Tensor sfq, Tensor sfk, Tensor sfv, Tensor delta_s, int unpadded_k, float softmax_scale, bool causal, bool per_block_mean, bool bf16_output, Tensor! out, Tensor! softmax_lse, Tensor! semaphore) -> ()");
  ops.impl("quantize_q_fp4_nhd", torch::kCUDA, &quantize_q);
  ops.impl("quantize_k_fp4_nhd", torch::kCUDA, &quantize_k);
  ops.impl("quantize_v_fp4_nhd", torch::kCUDA, &quantize_v);
  ops.impl("quantize_q_fp4_centered_nhd", torch::kCUDA, &quantize_q_centered);
  ops.impl("quantize_k_fp4_centered_nhd", torch::kCUDA, &quantize_k_centered);
  ops.impl("blockscaled_fp4_attention_static", torch::kCUDA, &attention_static);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
