#include <torch/all.h>
#include <torch/library.h>

#include <limits>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "nvfp4_sf_reshape_sm120.cuh"
#include "registration.h"
#include "torch_binding.h"

namespace {

void check_uint8_cuda_contiguous(torch::Tensor const& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kUInt8,
              name, " must have dtype torch.uint8");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
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

}  // namespace

void nvfp4_sf_linear_to_swizzled(
    torch::Tensor const& scales,
    torch::Tensor& out,
    int64_t D,
    bool is_sfb) {
  check_uint8_cuda_contiguous(scales, "scales");
  check_uint8_cuda_contiguous(out, "out");
  TORCH_CHECK(scales.dim() == 2, "scales must have shape (rows, D / 16)");
  TORCH_CHECK(scales.numel() > 0, "scales must be non-empty");
  TORCH_CHECK(D > 0 && D % 16 == 0, "D must be positive and divisible by 16");
  TORCH_CHECK(scales.size(1) == D / 16,
              "scales.shape[1] must equal D / 16");
  const int64_t rows64 = scales.size(0);
  const int64_t required = swizzled_bytes(rows64, D);
  TORCH_CHECK(out.numel() >= required,
              "out must contain at least nvfp4_sf_swizzled_bytes(rows, D) bytes");
  TORCH_CHECK(scales.get_device() == out.get_device(),
              "scales and out must be on the same CUDA device");

#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(scales.device());
  auto stream = at::cuda::getCurrentCUDAStream(scales.get_device()).stream();
  flash_rt::fp4::nvfp4_sf_linear_to_swizzled(
      scales.data_ptr(),
      out.data_ptr(),
      checked_int(rows64, "rows"),
      checked_int(D, "D"),
      is_sfb,
      stream);
#else
  TORCH_CHECK(false, "flashrt-nvfp4 was not built with CUDA support");
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("nvfp4_sf_linear_to_swizzled("
          "Tensor scales, Tensor! out, int D, bool is_sfb=False) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("nvfp4_sf_linear_to_swizzled",
           torch::kCUDA,
           &nvfp4_sf_linear_to_swizzled);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
