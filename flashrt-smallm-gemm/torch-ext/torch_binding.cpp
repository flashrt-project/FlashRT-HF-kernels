#include <torch/all.h>
#include <torch/library.h>

#include <limits>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "fp4_w4a4_matvec_sm120.cuh"
#include "registration.h"
#include "torch_binding.h"

namespace {

void check_cuda_contiguous(torch::Tensor const& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_uint8_cuda_contiguous(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kUInt8,
              name, " must have dtype torch.uint8");
}

void check_bf16_cuda_contiguous(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16,
              name, " must have dtype torch.bfloat16");
}

int checked_int(int64_t value, const char* name) {
  TORCH_CHECK(value >= 0 && value <= std::numeric_limits<int>::max(),
              name, " must fit in non-negative int");
  return static_cast<int>(value);
}

int64_t swizzled_bytes(int64_t rows, int64_t D) {
  const int64_t n_blocks = D / 16;
  const int64_t n_row_super = (rows + 127) / 128;
  const int64_t n_col_super = (n_blocks + 3) / 4;
  return n_row_super * n_col_super * 512;
}

void check_same_device(torch::Tensor const& a, torch::Tensor const& b,
                       const char* b_name) {
  TORCH_CHECK(a.get_device() == b.get_device(),
              "a_packed and ", b_name, " must be on the same CUDA device");
}

}  // namespace

void nvfp4_w4a4_decode_matvec_bf16out(
    torch::Tensor const& a_packed,
    torch::Tensor const& b_packed,
    torch::Tensor const& sfa,
    torch::Tensor const& sfb,
    torch::Tensor& out,
    double alpha) {
  check_uint8_cuda_contiguous(a_packed, "a_packed");
  check_uint8_cuda_contiguous(b_packed, "b_packed");
  check_uint8_cuda_contiguous(sfa, "sfa");
  check_uint8_cuda_contiguous(sfb, "sfb");
  check_bf16_cuda_contiguous(out, "out");

  TORCH_CHECK(b_packed.dim() == 2, "b_packed must have shape (N, K / 2)");
  TORCH_CHECK(a_packed.dim() == 1 ||
                  (a_packed.dim() == 2 && a_packed.size(0) == 1),
              "a_packed must have shape (K / 2,) or (1, K / 2)");
  const int64_t n64 = b_packed.size(0);
  const int64_t k_half64 = b_packed.size(1);
  TORCH_CHECK(n64 > 0, "N must be positive");
  TORCH_CHECK(k_half64 > 0, "K must be positive");
  TORCH_CHECK(a_packed.numel() == k_half64,
              "a_packed must contain K / 2 bytes");
  const int64_t k64 = k_half64 * 2;
  TORCH_CHECK(k64 == 4096 || k64 == 12288,
              "only K=4096 and K=12288 are supported by this decode matvec");
  TORCH_CHECK(out.numel() >= n64, "out must contain at least N BF16 values");

  const int64_t sfa_required = swizzled_bytes(1, k64);
  const int64_t sfb_required = swizzled_bytes(n64, k64);
  TORCH_CHECK(sfa.numel() >= sfa_required,
              "sfa must contain the swizzled bytes for one activation row");
  TORCH_CHECK(sfb.numel() >= sfb_required,
              "sfb must contain the swizzled bytes for N weight rows");
  check_same_device(a_packed, b_packed, "b_packed");
  check_same_device(a_packed, sfa, "sfa");
  check_same_device(a_packed, sfb, "sfb");
  check_same_device(a_packed, out, "out");

#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(a_packed.device());
  auto stream = at::cuda::getCurrentCUDAStream(a_packed.get_device()).stream();
  const int rc = flash_rt::gemm::fp4_w4a4_matvec_sm120_bf16out(
      a_packed.data_ptr(),
      b_packed.data_ptr(),
      out.data_ptr(),
      checked_int(n64, "N"),
      checked_int(k64, "K"),
      sfa.data_ptr(),
      sfb.data_ptr(),
      static_cast<float>(alpha),
      stream);
  TORCH_CHECK(rc == 0, "fp4_w4a4_matvec_sm120_bf16out failed with rc=", rc);
#else
  TORCH_CHECK(false, "flashrt-smallm-gemm was not built with CUDA support");
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("nvfp4_w4a4_decode_matvec_bf16out("
          "Tensor a_packed, Tensor b_packed, Tensor sfa, Tensor sfb, "
          "Tensor! out, float alpha=1.0) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("nvfp4_w4a4_decode_matvec_bf16out",
           torch::kCUDA,
           &nvfp4_w4a4_decode_matvec_bf16out);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
