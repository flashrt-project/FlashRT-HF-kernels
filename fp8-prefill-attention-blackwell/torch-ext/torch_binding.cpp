// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "fmha_fp8_causal_gqa_sm120.cuh"
#include "registration.h"
#include "torch_binding.h"

namespace {

void check_fp8_nhd(torch::Tensor const &tensor, const char *name, int heads) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == c10::ScalarType::Float8_e4m3fn, name,
              " must have dtype torch.float8_e4m3fn");
  TORCH_CHECK(tensor.dim() == 3, name,
              " must have shape (sequence, heads, 128)");
  TORCH_CHECK(tensor.size(1) == heads && tensor.size(2) == 128, name,
              " has an unsupported head layout");
}

} // namespace

void fp8_causal_gqa_attention_bf16_out(torch::Tensor const &query,
                                       torch::Tensor const &key,
                                       torch::Tensor const &value,
                                       double softmax_scale,
                                       torch::Tensor &output) {
  check_fp8_nhd(query, "query", 32);
  check_fp8_nhd(key, "key", 8);
  check_fp8_nhd(value, "value", 8);
  TORCH_CHECK(key.sizes() == value.sizes(),
              "key and value must have the same shape");
  TORCH_CHECK(query.size(0) == key.size(0),
              "causal self-attention requires equal Q and KV lengths");
  TORCH_CHECK(query.size(0) >= 256 && query.size(0) % 128 == 0,
              "sequence length must be a multiple of 128 and at least 256");
  TORCH_CHECK(output.is_cuda() && output.is_contiguous(),
              "output must be contiguous CUDA");
  TORCH_CHECK(output.scalar_type() == torch::kBFloat16,
              "output must have dtype torch.bfloat16");
  TORCH_CHECK(output.sizes() == query.sizes(),
              "output must have the same shape as query");
  TORCH_CHECK(query.get_device() == key.get_device() &&
                  query.get_device() == value.get_device() &&
                  query.get_device() == output.get_device(),
              "all tensors must be on the same CUDA device");
  TORCH_CHECK(softmax_scale > 0.0, "softmax_scale must be positive");

#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(query.device());
  auto stream = at::cuda::getCurrentCUDAStream(query.get_device()).stream();
  int rc = flash_rt::attention::fmha_fp8_causal_gqa_nhd_d128(
      query.data_ptr(), key.data_ptr(), value.data_ptr(), output.data_ptr(),
      static_cast<int>(query.size(0)), static_cast<int>(key.size(0)), 32, 8,
      static_cast<float>(softmax_scale), stream);
  TORCH_CHECK(rc == 0, "FP8 causal GQA attention failed with rc=", rc);
#else
  TORCH_CHECK(
      false, "fp8-prefill-attention-blackwell was not built with CUDA support");
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("fp8_causal_gqa_attention_bf16_out(Tensor query, Tensor key, Tensor "
          "value, float softmax_scale, Tensor! output) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("fp8_causal_gqa_attention_bf16_out", torch::kCUDA,
           &fp8_causal_gqa_attention_bf16_out);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
