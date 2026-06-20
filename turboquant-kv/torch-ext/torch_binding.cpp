// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "registration.h"
#include "torch_binding.h"
#include "turboquant_kv.cuh"

namespace {

void check_cuda_contiguous(torch::Tensor const& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_uint8(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kUInt8,
              name, " must have dtype torch.uint8");
}

void check_bf16(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16,
              name, " must have dtype torch.bfloat16");
}

void check_fp16(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16,
              name, " must have dtype torch.float16");
}

void check_fp32(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32,
              name, " must have dtype torch.float32");
}

void check_same_device(torch::Tensor const& a,
                       torch::Tensor const& b,
                       const char* a_name,
                       const char* b_name) {
  TORCH_CHECK(a.get_device() == b.get_device(),
              a_name, " and ", b_name, " must be on the same CUDA device");
}

int checked_bits(int64_t bits, const char* name) {
  TORCH_CHECK(bits == 2 || bits == 3 || bits == 4,
              name, " must be 2, 3, or 4");
  return static_cast<int>(bits);
}

int64_t check_packed_common(torch::Tensor const& k_idx_packed,
                            torch::Tensor const& k_qjl_packed,
                            torch::Tensor const& v_idx_packed,
                            torch::Tensor const& cb_k_mse,
                            torch::Tensor const& cb_v,
                            int64_t b_k_mse,
                            int64_t b_v) {
  check_uint8(k_idx_packed, "k_idx_packed");
  check_uint8(k_qjl_packed, "k_qjl_packed");
  check_uint8(v_idx_packed, "v_idx_packed");
  check_fp32(cb_k_mse, "cb_k_mse");
  check_fp32(cb_v, "cb_v");
  TORCH_CHECK(k_idx_packed.dim() == 2 && k_idx_packed.size(1) == 128,
              "k_idx_packed must have shape (M, 128)");
  const int64_t m = k_idx_packed.size(0);
  TORCH_CHECK(m > 0, "M must be positive");
  TORCH_CHECK(k_qjl_packed.sizes() == torch::IntArrayRef({m, 32}),
              "k_qjl_packed must have shape (M, 32)");
  TORCH_CHECK(v_idx_packed.sizes() == torch::IntArrayRef({m, 128}),
              "v_idx_packed must have shape (M, 128)");
  TORCH_CHECK(cb_k_mse.numel() >= (1LL << b_k_mse),
              "cb_k_mse must have at least 2**b_k_mse elements");
  TORCH_CHECK(cb_v.numel() >= (1LL << b_v),
              "cb_v must have at least 2**b_v elements");
  check_same_device(k_idx_packed, k_qjl_packed, "k_idx_packed", "k_qjl_packed");
  check_same_device(k_idx_packed, v_idx_packed, "k_idx_packed", "v_idx_packed");
  check_same_device(k_idx_packed, cb_k_mse, "k_idx_packed", "cb_k_mse");
  check_same_device(k_idx_packed, cb_v, "k_idx_packed", "cb_v");
  return m;
}

}  // namespace

void unpack_packed_bf16(
    torch::Tensor const& k_idx_packed,
    torch::Tensor const& k_qjl_packed,
    torch::Tensor const& v_idx_packed,
    torch::Tensor const& cb_k_mse,
    torch::Tensor const& cb_v,
    int64_t b_k_mse64,
    int64_t b_v64,
    torch::Tensor& y_k,
    torch::Tensor& qjl_bf,
    torch::Tensor& y_v) {
  const int b_k_mse = checked_bits(b_k_mse64, "b_k_mse");
  const int b_v = checked_bits(b_v64, "b_v");
  const int64_t m = check_packed_common(k_idx_packed, k_qjl_packed, v_idx_packed, cb_k_mse, cb_v, b_k_mse, b_v);
  check_bf16(y_k, "y_k");
  check_bf16(qjl_bf, "qjl_bf");
  check_bf16(y_v, "y_v");
  TORCH_CHECK(y_k.sizes() == torch::IntArrayRef({m, 256}), "y_k must have shape (M, 256)");
  TORCH_CHECK(qjl_bf.sizes() == y_k.sizes(), "qjl_bf must have shape (M, 256)");
  TORCH_CHECK(y_v.sizes() == y_k.sizes(), "y_v must have shape (M, 256)");
  check_same_device(k_idx_packed, y_k, "k_idx_packed", "y_k");
  check_same_device(k_idx_packed, qjl_bf, "k_idx_packed", "qjl_bf");
  check_same_device(k_idx_packed, y_v, "k_idx_packed", "y_v");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(k_idx_packed.device());
  auto stream = at::cuda::getCurrentCUDAStream(k_idx_packed.get_device()).stream();
  flash_rt::turboquant_kv::unpack_packed_bf16(
      k_idx_packed.data_ptr(), k_qjl_packed.data_ptr(), v_idx_packed.data_ptr(),
      cb_k_mse.data_ptr(), cb_v.data_ptr(), y_k.data_ptr(), qjl_bf.data_ptr(),
      y_v.data_ptr(), static_cast<int>(m), b_k_mse, b_v, stream);
#else
  TORCH_CHECK(false, "turboquant-kv was not built with CUDA support");
#endif
}

void unpack_packed_mixed(
    torch::Tensor const& k_idx_packed,
    torch::Tensor const& k_qjl_packed,
    torch::Tensor const& v_idx_packed,
    torch::Tensor const& cb_k_mse,
    torch::Tensor const& cb_v,
    int64_t b_k_mse64,
    int64_t b_v64,
    torch::Tensor& y_k,
    torch::Tensor& qjl_f,
    torch::Tensor& y_v) {
  const int b_k_mse = checked_bits(b_k_mse64, "b_k_mse");
  const int b_v = checked_bits(b_v64, "b_v");
  const int64_t m = check_packed_common(k_idx_packed, k_qjl_packed, v_idx_packed, cb_k_mse, cb_v, b_k_mse, b_v);
  check_bf16(y_k, "y_k");
  check_fp32(qjl_f, "qjl_f");
  check_bf16(y_v, "y_v");
  TORCH_CHECK(y_k.sizes() == torch::IntArrayRef({m, 256}), "y_k must have shape (M, 256)");
  TORCH_CHECK(qjl_f.sizes() == y_k.sizes(), "qjl_f must have shape (M, 256)");
  TORCH_CHECK(y_v.sizes() == y_k.sizes(), "y_v must have shape (M, 256)");
  check_same_device(k_idx_packed, y_k, "k_idx_packed", "y_k");
  check_same_device(k_idx_packed, qjl_f, "k_idx_packed", "qjl_f");
  check_same_device(k_idx_packed, y_v, "k_idx_packed", "y_v");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(k_idx_packed.device());
  auto stream = at::cuda::getCurrentCUDAStream(k_idx_packed.get_device()).stream();
  flash_rt::turboquant_kv::unpack_packed_mixed(
      k_idx_packed.data_ptr(), k_qjl_packed.data_ptr(), v_idx_packed.data_ptr(),
      cb_k_mse.data_ptr(), cb_v.data_ptr(), y_k.data_ptr(), qjl_f.data_ptr(),
      y_v.data_ptr(), static_cast<int>(m), b_k_mse, b_v, stream);
#else
  TORCH_CHECK(false, "turboquant-kv was not built with CUDA support");
#endif
}

void combine_kv_bf16(
    torch::Tensor const& k_mse,
    torch::Tensor const& k_qjl,
    torch::Tensor const& v_unit,
    torch::Tensor const& k_norm,
    torch::Tensor const& k_rnorm,
    torch::Tensor const& v_norm,
    double coef,
    torch::Tensor& k_out,
    torch::Tensor& v_out) {
  check_bf16(k_mse, "k_mse");
  check_bf16(k_qjl, "k_qjl");
  check_bf16(v_unit, "v_unit");
  check_fp16(k_norm, "k_norm");
  check_fp16(k_rnorm, "k_rnorm");
  check_fp16(v_norm, "v_norm");
  check_bf16(k_out, "k_out");
  check_bf16(v_out, "v_out");
  TORCH_CHECK(k_mse.dim() == 2 && k_mse.size(1) == 256, "k_mse must have shape (M, 256)");
  const int64_t m = k_mse.size(0);
  TORCH_CHECK(k_qjl.sizes() == k_mse.sizes() && v_unit.sizes() == k_mse.sizes(),
              "k_qjl and v_unit must have shape (M, 256)");
  TORCH_CHECK(k_norm.sizes() == torch::IntArrayRef({m}) &&
              k_rnorm.sizes() == torch::IntArrayRef({m}) &&
              v_norm.sizes() == torch::IntArrayRef({m}),
              "norm tensors must have shape (M,)");
  TORCH_CHECK(k_out.sizes() == k_mse.sizes() && v_out.sizes() == k_mse.sizes(),
              "outputs must have shape (M, 256)");
  check_same_device(k_mse, k_qjl, "k_mse", "k_qjl");
  check_same_device(k_mse, v_unit, "k_mse", "v_unit");
  check_same_device(k_mse, k_norm, "k_mse", "k_norm");
  check_same_device(k_mse, k_rnorm, "k_mse", "k_rnorm");
  check_same_device(k_mse, v_norm, "k_mse", "v_norm");
  check_same_device(k_mse, k_out, "k_mse", "k_out");
  check_same_device(k_mse, v_out, "k_mse", "v_out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(k_mse.device());
  auto stream = at::cuda::getCurrentCUDAStream(k_mse.get_device()).stream();
  flash_rt::turboquant_kv::combine_kv_bf16(
      k_mse.data_ptr(), k_qjl.data_ptr(), v_unit.data_ptr(),
      k_norm.data_ptr(), k_rnorm.data_ptr(), v_norm.data_ptr(),
      k_out.data_ptr(), v_out.data_ptr(),
      static_cast<int>(m), static_cast<float>(coef), stream);
#else
  TORCH_CHECK(false, "turboquant-kv was not built with CUDA support");
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("unpack_packed_bf16(Tensor k_idx_packed, Tensor k_qjl_packed, Tensor v_idx_packed, Tensor cb_k_mse, Tensor cb_v, int b_k_mse, int b_v, Tensor! y_k, Tensor! qjl_bf, Tensor! y_v) -> ()");
  ops.def("unpack_packed_mixed(Tensor k_idx_packed, Tensor k_qjl_packed, Tensor v_idx_packed, Tensor cb_k_mse, Tensor cb_v, int b_k_mse, int b_v, Tensor! y_k, Tensor! qjl_f, Tensor! y_v) -> ()");
  ops.def("combine_kv_bf16(Tensor k_mse, Tensor k_qjl, Tensor v_unit, Tensor k_norm, Tensor k_rnorm, Tensor v_norm, float coef, Tensor! k_out, Tensor! v_out) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("unpack_packed_bf16", torch::kCUDA, &unpack_packed_bf16);
  ops.impl("unpack_packed_mixed", torch::kCUDA, &unpack_packed_mixed);
  ops.impl("combine_kv_bf16", torch::kCUDA, &combine_kv_bf16);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
