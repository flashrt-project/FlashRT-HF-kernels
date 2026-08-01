// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "delayed_codebook_kernels.cuh"
#include "registration.h"
#include "torch_binding.h"

namespace {

void check_cuda_contiguous(torch::Tensor const& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_contract(torch::Tensor const& logits,
                    torch::Tensor const& codebook,
                    int64_t delay, int64_t boc,
                    torch::Tensor const& codes,
                    torch::Tensor const& embedding) {
  check_cuda_contiguous(logits, "logits");
  check_cuda_contiguous(codebook, "codebook");
  check_cuda_contiguous(codes, "codes");
  check_cuda_contiguous(embedding, "embedding");
  TORCH_CHECK(logits.scalar_type() == torch::kBFloat16,
              "logits must have dtype torch.bfloat16");
  TORCH_CHECK(codebook.scalar_type() == torch::kBFloat16,
              "codebook must have dtype torch.bfloat16");
  TORCH_CHECK(codes.scalar_type() == torch::kInt64,
              "codes must have dtype torch.int64");
  TORCH_CHECK(embedding.scalar_type() == torch::kBFloat16,
              "embedding must have dtype torch.bfloat16");
  TORCH_CHECK(logits.dim() == 2,
              "logits must have shape (num_codebooks, codebook_vocab)");
  TORCH_CHECK(codebook.dim() == 3,
              "codebook must have shape (num_codebooks, codebook_vocab, hidden)");
  TORCH_CHECK(logits.size(0) > 0 && logits.size(1) > 0 &&
                  codebook.size(2) > 0,
              "all codebook dimensions must be positive");
  TORCH_CHECK(codebook.size(0) == logits.size(0) &&
                  codebook.size(1) == logits.size(1),
              "codebook leading dimensions must match logits");
  TORCH_CHECK(codes.sizes() == torch::IntArrayRef({logits.size(0)}),
              "codes must have shape (num_codebooks,)");
  TORCH_CHECK(embedding.sizes() ==
                  torch::IntArrayRef({codebook.size(2)}),
              "embedding must have shape (hidden,)");
  TORCH_CHECK(delay >= 0 && delay <= logits.size(0),
              "delay must be in [0, num_codebooks]");
  TORCH_CHECK(boc >= 0 && boc < logits.size(1),
              "boc must be a valid codebook index");
  TORCH_CHECK(logits.get_device() == codebook.get_device() &&
                  logits.get_device() == codes.get_device() &&
                  logits.get_device() == embedding.get_device(),
              "all tensors must be on the same CUDA device");
}

}  // namespace

void delayed_codebook_argmax_embed_bf16(
    torch::Tensor const& logits, torch::Tensor const& codebook,
    int64_t delay, int64_t boc, torch::Tensor& codes,
    torch::Tensor& embedding) {
  check_contract(logits, codebook, delay, boc, codes, embedding);
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard guard(logits.device());
  auto stream = at::cuda::getCurrentCUDAStream(logits.get_device()).stream();
  flash_rt::kernels::delayed_codebook_argmax_embed_bf16(
      static_cast<const __nv_bfloat16*>(logits.data_ptr()),
      static_cast<const __nv_bfloat16*>(codebook.data_ptr()),
      codes.data_ptr<int64_t>(),
      static_cast<__nv_bfloat16*>(embedding.data_ptr()),
      static_cast<int>(logits.size(0)), static_cast<int>(logits.size(1)),
      static_cast<int>(codebook.size(2)), static_cast<int>(delay),
      static_cast<int>(boc), stream);
#else
  TORCH_CHECK(false, "audio-codebook-primitives was not built with CUDA support");
#endif
}

void delayed_codebook_sample_embed_bf16(
    torch::Tensor const& logits, torch::Tensor const& codebook,
    int64_t delay, int64_t boc, double temperature,
    int64_t seed, int64_t step, torch::Tensor& codes,
    torch::Tensor& embedding) {
  check_contract(logits, codebook, delay, boc, codes, embedding);
  TORCH_CHECK(temperature > 0.0,
              "temperature must be strictly positive");
  TORCH_CHECK(step >= 0, "step must be non-negative");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard guard(logits.device());
  auto stream = at::cuda::getCurrentCUDAStream(logits.get_device()).stream();
  flash_rt::kernels::delayed_codebook_sample_embed_bf16(
      static_cast<const __nv_bfloat16*>(logits.data_ptr()),
      static_cast<const __nv_bfloat16*>(codebook.data_ptr()),
      codes.data_ptr<int64_t>(),
      static_cast<__nv_bfloat16*>(embedding.data_ptr()),
      static_cast<int>(logits.size(0)), static_cast<int>(logits.size(1)),
      static_cast<int>(codebook.size(2)), static_cast<int>(delay),
      static_cast<int>(boc), static_cast<float>(temperature),
      static_cast<uint64_t>(seed), static_cast<uint64_t>(step), stream);
#else
  TORCH_CHECK(false, "audio-codebook-primitives was not built with CUDA support");
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("delayed_codebook_argmax_embed_bf16(Tensor logits, Tensor codebook, int delay, int boc, Tensor! codes, Tensor! embedding) -> ()");
  ops.def("delayed_codebook_sample_embed_bf16(Tensor logits, Tensor codebook, int delay, int boc, float temperature, int seed, int step, Tensor! codes, Tensor! embedding) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("delayed_codebook_argmax_embed_bf16", torch::kCUDA,
           &delayed_codebook_argmax_embed_bf16);
  ops.impl("delayed_codebook_sample_embed_bf16", torch::kCUDA,
           &delayed_codebook_sample_embed_bf16);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
