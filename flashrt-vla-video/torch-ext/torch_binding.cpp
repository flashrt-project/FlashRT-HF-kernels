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

void check_fp32_cuda_contiguous(torch::Tensor const& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32,
              name, " must have dtype torch.float32");
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

void qkv_split_norm_rope_bf16(
    torch::Tensor const& packed_qkv,
    torch::Tensor const& norm_q_weight,
    torch::Tensor const& norm_k_weight,
    torch::Tensor const& freqs_re,
    torch::Tensor const& freqs_im,
    torch::Tensor& q_out,
    torch::Tensor& k_out,
    int64_t heads,
    int64_t head_dim,
    int64_t seq_len,
    double eps) {
  check_bf16_cuda_contiguous(packed_qkv, "packed_qkv");
  check_bf16_cuda_contiguous(norm_q_weight, "norm_q_weight");
  check_bf16_cuda_contiguous(norm_k_weight, "norm_k_weight");
  check_fp32_cuda_contiguous(freqs_re, "freqs_re");
  check_fp32_cuda_contiguous(freqs_im, "freqs_im");
  check_bf16_cuda_contiguous(q_out, "q_out");
  check_bf16_cuda_contiguous(k_out, "k_out");

  TORCH_CHECK(packed_qkv.dim() == 3,
              "packed_qkv must have shape (B, L, 3 * heads * head_dim)");
  TORCH_CHECK(heads > 0, "heads must be positive");
  TORCH_CHECK(head_dim > 0 && head_dim % 2 == 0,
              "head_dim must be positive and even");
  TORCH_CHECK(heads <= std::numeric_limits<int>::max(),
              "heads must fit in int");
  TORCH_CHECK(head_dim <= std::numeric_limits<int>::max(),
              "head_dim must fit in int");
  TORCH_CHECK(seq_len >= 0 && seq_len <= std::numeric_limits<int>::max(),
              "seq_len must fit in non-negative int");
  const auto batch = packed_qkv.size(0);
  const auto tokens = packed_qkv.size(1);
  const auto dim = heads * head_dim;
  TORCH_CHECK(batch > 0 && tokens > 0,
              "packed_qkv batch and token dimensions must be non-empty");
  TORCH_CHECK(packed_qkv.size(2) == 3 * dim,
              "packed_qkv last dimension must be 3 * heads * head_dim");
  TORCH_CHECK(norm_q_weight.dim() == 1 && norm_q_weight.size(0) == dim,
              "norm_q_weight must have shape (heads * head_dim,)");
  TORCH_CHECK(norm_k_weight.dim() == 1 && norm_k_weight.size(0) == dim,
              "norm_k_weight must have shape (heads * head_dim,)");
  TORCH_CHECK(freqs_re.dim() == 2 && freqs_re.size(1) == head_dim / 2,
              "freqs_re must have shape (seq_len_table, head_dim / 2)");
  TORCH_CHECK(freqs_im.sizes() == freqs_re.sizes(),
              "freqs_im must have the same shape as freqs_re");
  TORCH_CHECK(seq_len <= freqs_re.size(0),
              "seq_len cannot exceed freqs_re/freqs_im length");
  TORCH_CHECK(q_out.dim() == 4 && q_out.size(0) == batch &&
                  q_out.size(1) == tokens && q_out.size(2) == heads &&
                  q_out.size(3) == head_dim,
              "q_out must have shape (B, L, heads, head_dim)");
  TORCH_CHECK(k_out.dim() == 4 && k_out.size(0) == batch &&
                  k_out.size(1) == tokens && k_out.size(2) == heads &&
                  k_out.size(3) == head_dim,
              "k_out must have the same shape as q_out");
  TORCH_CHECK(packed_qkv.get_device() == norm_q_weight.get_device(),
              "packed_qkv and norm_q_weight must be on the same CUDA device");
  TORCH_CHECK(packed_qkv.get_device() == norm_k_weight.get_device(),
              "packed_qkv and norm_k_weight must be on the same CUDA device");
  TORCH_CHECK(packed_qkv.get_device() == freqs_re.get_device(),
              "packed_qkv and freqs_re must be on the same CUDA device");
  TORCH_CHECK(packed_qkv.get_device() == freqs_im.get_device(),
              "packed_qkv and freqs_im must be on the same CUDA device");
  TORCH_CHECK(packed_qkv.get_device() == q_out.get_device(),
              "packed_qkv and q_out must be on the same CUDA device");
  TORCH_CHECK(packed_qkv.get_device() == k_out.get_device(),
              "packed_qkv and k_out must be on the same CUDA device");
  TORCH_CHECK(batch <= std::numeric_limits<int>::max(),
              "batch must fit in int");
  TORCH_CHECK(tokens <= std::numeric_limits<int>::max(),
              "tokens must fit in int");

#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(packed_qkv.device());
  auto stream =
      at::cuda::getCurrentCUDAStream(packed_qkv.get_device()).stream();
  flash_rt::vla_video::qkv_split_norm_rope_bf16(
      packed_qkv.data_ptr(),
      norm_q_weight.data_ptr(),
      norm_k_weight.data_ptr(),
      freqs_re.data_ptr(),
      freqs_im.data_ptr(),
      q_out.data_ptr(),
      k_out.data_ptr(),
      static_cast<int>(batch),
      static_cast<int>(tokens),
      static_cast<int>(heads),
      static_cast<int>(head_dim),
      static_cast<int>(seq_len),
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
  ops.def("qkv_split_norm_rope_bf16("
          "Tensor packed_qkv, Tensor norm_q_weight, Tensor norm_k_weight, "
          "Tensor freqs_re, Tensor freqs_im, Tensor! q_out, Tensor! k_out, "
          "int heads, int head_dim, int seq_len, float eps=1e-6) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("q_norm_rope_bf16", torch::kCUDA, &q_norm_rope_bf16);
  ops.impl("k_norm_rope_v_cache_bf16",
           torch::kCUDA,
           &k_norm_rope_v_cache_bf16);
  ops.impl("qkv_split_norm_rope_bf16",
           torch::kCUDA,
           &qkv_split_norm_rope_bf16);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
