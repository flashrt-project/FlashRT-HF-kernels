// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "kernels/nexn2_gdn_seq.cuh"
#include "registration.h"
#include "torch_binding.h"

namespace {

void check_cuda_contiguous(torch::Tensor const& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_bf16(torch::Tensor const& t, const char* name) {
  check_cuda_contiguous(t, name);
  TORCH_CHECK(t.scalar_type() == torch::kBFloat16, name, " must be torch.bfloat16");
}

}  // namespace

void gated_delta_recurrent_seq_bf16(torch::Tensor const& q,
                                    torch::Tensor const& k,
                                    torch::Tensor const& v,
                                    torch::Tensor const& g,
                                    torch::Tensor const& beta,
                                    torch::Tensor& state,
                                    torch::Tensor& out,
                                    bool use_qk_l2norm) {
  check_bf16(q, "q");
  check_bf16(k, "k");
  check_bf16(v, "v");
  check_bf16(g, "g");
  check_bf16(beta, "beta");
  check_bf16(state, "state");
  check_bf16(out, "out");
  TORCH_CHECK(q.dim() == 3, "q must have shape (S,H,D)");
  TORCH_CHECK(k.sizes() == q.sizes() && v.sizes() == q.sizes(), "k/v shape mismatch");
  TORCH_CHECK(g.sizes() == torch::IntArrayRef({q.size(0), q.size(1)}), "g shape mismatch");
  TORCH_CHECK(beta.sizes() == g.sizes(), "beta shape mismatch");
  TORCH_CHECK(state.sizes() == torch::IntArrayRef({q.size(1), q.size(2), q.size(2)}),
              "state must have shape (H,D,D)");
  TORCH_CHECK(out.sizes() == q.sizes(), "out shape mismatch");
  TORCH_CHECK(q.size(2) == 128, "head_dim must be 128");
#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device()).stream();
  const int rc = flash_rt::kernels::nexn2_gdn_recurrent_seq_bf16(
      q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(), beta.data_ptr(),
      state.data_ptr(), out.data_ptr(),
      static_cast<int>(q.size(0)), static_cast<int>(q.size(1)),
      static_cast<int>(q.size(2)), use_qk_l2norm, stream);
  TORCH_CHECK(rc == 0, "gated_delta_recurrent_seq_bf16 failed with rc=", rc);
#else
  TORCH_CHECK(false, "linear-attention-seq-state was not built with CUDA support");
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("gated_delta_recurrent_seq_bf16(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor! state, Tensor! out, bool use_qk_l2norm=False) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("gated_delta_recurrent_seq_bf16", torch::kCUDA, &gated_delta_recurrent_seq_bf16);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
