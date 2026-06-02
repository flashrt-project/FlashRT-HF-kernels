#include <torch/all.h>
#include <torch/library.h>

#include <limits>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "q_norm_rope_bf16.cuh"
#include "registration.h"
#include "torch_binding.h"

namespace {

constexpr int kHeadDim = 128;
constexpr int kHalfDim = kHeadDim / 2;

void check_bf16_cuda_contiguous(torch::Tensor const& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16,
              name, " must have dtype torch.bfloat16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

int checked_rows(torch::Tensor const& tensor, const char* name) {
  TORCH_CHECK(tensor.dim() >= 2, name, " must have at least 2 dimensions");
  TORCH_CHECK(tensor.size(-1) == kHeadDim,
              name, " last dimension must be 128");
  TORCH_CHECK(tensor.numel() > 0, name, " must be non-empty");
  const auto rows64 = tensor.numel() / kHeadDim;
  TORCH_CHECK(rows64 <= std::numeric_limits<int>::max(),
              name, " flattened rows must fit in int");
  return static_cast<int>(rows64);
}

void check_common(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    torch::Tensor const& cos,
    torch::Tensor const& sin,
    torch::Tensor const& out) {
  check_bf16_cuda_contiguous(input, "input");
  check_bf16_cuda_contiguous(weight, "weight");
  check_bf16_cuda_contiguous(cos, "cos");
  check_bf16_cuda_contiguous(sin, "sin");
  check_bf16_cuda_contiguous(out, "out");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == kHeadDim,
              "weight must have shape (128,)");
  TORCH_CHECK(cos.dim() == 1 && cos.size(0) == kHalfDim,
              "cos must have shape (64,)");
  TORCH_CHECK(sin.dim() == 1 && sin.size(0) == kHalfDim,
              "sin must have shape (64,)");
  TORCH_CHECK(out.sizes() == input.sizes(),
              "out must have the same shape as input");
  TORCH_CHECK(input.get_device() == weight.get_device(),
              "input and weight must be on the same CUDA device");
  TORCH_CHECK(input.get_device() == cos.get_device(),
              "input and cos must be on the same CUDA device");
  TORCH_CHECK(input.get_device() == sin.get_device(),
              "input and sin must be on the same CUDA device");
  TORCH_CHECK(input.get_device() == out.get_device(),
              "input and out must be on the same CUDA device");
}

}  // namespace

void q_norm_rope_bf16(
    torch::Tensor const& q,
    torch::Tensor const& weight,
    torch::Tensor const& cos,
    torch::Tensor const& sin,
    torch::Tensor& out,
    double eps) {
  check_common(q, weight, cos, sin, out);
  const int rows = checked_rows(q, "q");

#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device()).stream();
  flash_rt::vla_video::q_norm_rope_bf16(
      q.data_ptr(),
      weight.data_ptr(),
      cos.data_ptr(),
      sin.data_ptr(),
      out.data_ptr(),
      rows,
      static_cast<float>(eps),
      stream);
#else
  TORCH_CHECK(false, "flashrt-vla-video was not built with CUDA support");
#endif
}

void k_norm_rope_v_cache_bf16(
    torch::Tensor const& k,
    torch::Tensor const& v,
    torch::Tensor const& weight,
    torch::Tensor const& cos,
    torch::Tensor const& sin,
    torch::Tensor& k_out,
    torch::Tensor& v_out,
    double eps) {
  check_common(k, weight, cos, sin, k_out);
  check_bf16_cuda_contiguous(v, "v");
  check_bf16_cuda_contiguous(v_out, "v_out");
  TORCH_CHECK(v.sizes() == k.sizes(), "v must have the same shape as k");
  TORCH_CHECK(v_out.sizes() == k.sizes(),
              "v_out must have the same shape as k");
  TORCH_CHECK(k.get_device() == v.get_device(),
              "k and v must be on the same CUDA device");
  TORCH_CHECK(k.get_device() == v_out.get_device(),
              "k and v_out must be on the same CUDA device");
  const int rows = checked_rows(k, "k");

#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(k.device());
  auto stream = at::cuda::getCurrentCUDAStream(k.get_device()).stream();
  flash_rt::vla_video::k_norm_rope_v_cache_bf16(
      k.data_ptr(),
      v.data_ptr(),
      weight.data_ptr(),
      cos.data_ptr(),
      sin.data_ptr(),
      k_out.data_ptr(),
      v_out.data_ptr(),
      rows,
      static_cast<float>(eps),
      stream);
#else
  TORCH_CHECK(false, "flashrt-vla-video was not built with CUDA support");
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("q_norm_rope_bf16("
          "Tensor q, Tensor weight, Tensor cos, Tensor sin, Tensor! out, "
          "float eps=1e-6) -> ()");
  ops.def("k_norm_rope_v_cache_bf16("
          "Tensor k, Tensor v, Tensor weight, Tensor cos, Tensor sin, "
          "Tensor! k_out, Tensor! v_out, float eps=1e-6) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("q_norm_rope_bf16", torch::kCUDA, &q_norm_rope_bf16);
  ops.impl("k_norm_rope_v_cache_bf16",
           torch::kCUDA,
           &k_norm_rope_v_cache_bf16);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
