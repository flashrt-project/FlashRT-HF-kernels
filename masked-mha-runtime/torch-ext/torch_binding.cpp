#include <torch/all.h>
#include <torch/library.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <limits>

#include "attention_mha_masked.cuh"
#include "registration.h"

namespace {

int checked_int(int64_t value, const char* name) {
  TORCH_CHECK(value > 0 && value <= std::numeric_limits<int>::max(),
              name, " must fit in a positive int");
  return static_cast<int>(value);
}

void check_qkv(torch::Tensor const& tensor, const char* name,
               c10::ScalarType dtype) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
  TORCH_CHECK(tensor.dim() == 3, name, " must have shape (S, H, D)");
  TORCH_CHECK(tensor.stride(2) == 1 && tensor.stride(1) == tensor.size(2),
              name, " must be contiguous within each token");
}

void masked_mha_forward_static(
    torch::Tensor const& q, torch::Tensor const& k, torch::Tensor const& v,
    torch::Tensor& logits, torch::Tensor& out, double scale) {
  TORCH_CHECK(q.scalar_type() == torch::kFloat16 ||
                  q.scalar_type() == torch::kBFloat16,
              "q must be FP16 or BF16");
  check_qkv(q, "q", q.scalar_type());
  check_qkv(k, "k", q.scalar_type());
  check_qkv(v, "v", q.scalar_type());
  TORCH_CHECK(q.size(1) == k.size(1) && q.size(1) == v.size(1) &&
                  q.size(2) == k.size(2) && q.size(2) == v.size(2) &&
                  k.size(0) == v.size(0),
              "q/k/v head shapes must match");
  TORCH_CHECK(q.get_device() == k.get_device() &&
                  q.get_device() == v.get_device(),
              "q/k/v must be on the same device");
  TORCH_CHECK(out.is_cuda() && out.is_contiguous() &&
                  out.scalar_type() == q.scalar_type() &&
                  out.sizes() == q.sizes(),
              "out must be contiguous and match q");
  TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == q.scalar_type() &&
                  logits.dim() == 3 && logits.size(0) == q.size(1) &&
                  logits.size(1) == q.size(0) &&
                  logits.size(2) >= k.size(0) && logits.stride(2) == 1,
              "logits must have shape (H, S_q, stride >= S_kv)");
  TORCH_CHECK(logits.get_device() == q.get_device() &&
                  out.get_device() == q.get_device(),
              "outputs must be on the q device");
  TORCH_CHECK(logits.stride(1) == logits.size(2) &&
                  logits.stride(0) == logits.size(1) * logits.size(2),
              "logits must use a dense padded row stride");

  c10::cuda::CUDAGuard guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device()).stream();
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  const int sq = checked_int(q.size(0), "S_q");
  const int sk = checked_int(k.size(0), "S_kv");
  const int heads = checked_int(q.size(1), "heads");
  const int dim = checked_int(q.size(2), "head_dim");

  if (q.scalar_type() == torch::kFloat16) {
    TORCH_CHECK(q.stride(0) == heads * dim &&
                    k.stride(0) == heads * dim &&
                    v.stride(0) == heads * dim,
                "FP16 q/k/v must be contiguous across tokens");
    attention_mha_fp16_masked(
        handle, static_cast<const __half*>(q.data_ptr()),
        static_cast<const __half*>(k.data_ptr()),
        static_cast<const __half*>(v.data_ptr()),
        static_cast<__half*>(logits.data_ptr()),
        static_cast<__half*>(out.data_ptr()), sq, sk, heads, dim,
        static_cast<float>(scale), stream);
  } else {
    TORCH_CHECK(q.stride(0) == k.stride(0) && q.stride(0) == v.stride(0),
                "BF16 q/k/v must share one token stride");
    attention_mha_bf16_masked(
        handle, static_cast<const __nv_bfloat16*>(q.data_ptr()),
        static_cast<const __nv_bfloat16*>(k.data_ptr()),
        static_cast<const __nv_bfloat16*>(v.data_ptr()),
        static_cast<__nv_bfloat16*>(logits.data_ptr()),
        static_cast<__nv_bfloat16*>(out.data_ptr()), sq, sk, heads, dim,
        static_cast<float>(scale), checked_int(logits.size(2), "logits stride"),
        checked_int(q.stride(0), "qkv token stride"), stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("forward_static(Tensor q, Tensor k, Tensor v, Tensor! logits, Tensor! out, float scale) -> ()");
  ops.impl("forward_static", torch::kCUDA, &masked_mha_forward_static);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
