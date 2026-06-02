#include <torch/all.h>
#include <torch/library.h>

#include <limits>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "registration.h"
#include "silu_mul_to_nvfp4_swizzled.cuh"
#include "torch_binding.h"

namespace {

void check_cuda_contiguous(torch::Tensor const& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_bf16_cuda_contiguous(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16,
              name, " must have dtype torch.bfloat16");
}

void check_uint8_cuda_contiguous(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kUInt8,
              name, " must have dtype torch.uint8");
}

int checked_int(int64_t value, const char* name) {
  TORCH_CHECK(value >= 0 && value <= std::numeric_limits<int>::max(),
              name, " must fit in non-negative int");
  return static_cast<int>(value);
}

int64_t swizzled_bytes(int64_t rows, int64_t cols) {
  const int64_t n_blocks = cols / 16;
  const int64_t n_row_super = (rows + 127) / 128;
  const int64_t n_col_super = (n_blocks + 3) / 4;
  return n_row_super * n_col_super * 512;
}

void check_outputs(torch::Tensor const& ref,
                   torch::Tensor const& packed,
                   torch::Tensor const& scales,
                   int64_t rows,
                   int64_t cols) {
  check_uint8_cuda_contiguous(packed, "packed");
  check_uint8_cuda_contiguous(scales, "scales");
  TORCH_CHECK(packed.numel() >= rows * (cols / 2),
              "packed must contain at least rows * cols / 2 bytes");
  TORCH_CHECK(scales.numel() >= swizzled_bytes(rows, cols),
              "scales must contain the swizzled scale-factor buffer");
  TORCH_CHECK(ref.get_device() == packed.get_device(),
              "input and packed must be on the same CUDA device");
  TORCH_CHECK(ref.get_device() == scales.get_device(),
              "input and scales must be on the same CUDA device");
}

}  // namespace

void silu_mul_quant_nvfp4_swizzled_bf16(
    torch::Tensor const& gate,
    torch::Tensor const& up,
    torch::Tensor& packed,
    torch::Tensor& scales) {
  check_bf16_cuda_contiguous(gate, "gate");
  check_bf16_cuda_contiguous(up, "up");
  TORCH_CHECK(gate.dim() == 2, "gate must have shape (rows, cols)");
  TORCH_CHECK(up.sizes() == gate.sizes(), "up must have the same shape as gate");
  const int64_t rows64 = gate.size(0);
  const int64_t cols64 = gate.size(1);
  TORCH_CHECK(rows64 > 0, "rows must be positive");
  TORCH_CHECK(cols64 > 0 && cols64 % 16 == 0,
              "cols must be positive and divisible by 16");
  TORCH_CHECK(gate.get_device() == up.get_device(),
              "gate and up must be on the same CUDA device");
  check_outputs(gate, packed, scales, rows64, cols64);

#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(gate.device());
  auto stream = at::cuda::getCurrentCUDAStream(gate.get_device()).stream();
  const int rc = flash_rt::kernels::silu_mul_to_nvfp4_swizzled_bf16(
      gate.data_ptr(),
      up.data_ptr(),
      packed.data_ptr(),
      scales.data_ptr(),
      checked_int(rows64, "rows"),
      checked_int(cols64, "cols"),
      stream);
  TORCH_CHECK(rc == 0, "silu_mul_to_nvfp4_swizzled_bf16 failed with rc=", rc);
#else
  TORCH_CHECK(false, "flashrt-fused-quant was not built with CUDA support");
#endif
}

void silu_mul_merged_quant_nvfp4_swizzled_bf16(
    torch::Tensor const& merged_gate_up,
    torch::Tensor& packed,
    torch::Tensor& scales) {
  check_bf16_cuda_contiguous(merged_gate_up, "merged_gate_up");
  TORCH_CHECK(merged_gate_up.dim() == 2,
              "merged_gate_up must have shape (rows, 2 * cols)");
  const int64_t rows64 = merged_gate_up.size(0);
  const int64_t merged_cols64 = merged_gate_up.size(1);
  TORCH_CHECK(rows64 > 0, "rows must be positive");
  TORCH_CHECK(merged_cols64 > 0 && merged_cols64 % 32 == 0,
              "merged_gate_up.shape[1] must be positive and divisible by 32");
  const int64_t cols64 = merged_cols64 / 2;
  check_outputs(merged_gate_up, packed, scales, rows64, cols64);

#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(merged_gate_up.device());
  auto stream = at::cuda::getCurrentCUDAStream(merged_gate_up.get_device()).stream();
  const int rc = flash_rt::kernels::silu_mul_merged_to_nvfp4_swizzled_bf16(
      merged_gate_up.data_ptr(),
      packed.data_ptr(),
      scales.data_ptr(),
      checked_int(rows64, "rows"),
      checked_int(cols64, "cols"),
      stream);
  TORCH_CHECK(rc == 0,
              "silu_mul_merged_to_nvfp4_swizzled_bf16 failed with rc=", rc);
#else
  TORCH_CHECK(false, "flashrt-fused-quant was not built with CUDA support");
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("silu_mul_quant_nvfp4_swizzled_bf16("
          "Tensor gate, Tensor up, Tensor! packed, Tensor! scales) -> ()");
  ops.def("silu_mul_merged_quant_nvfp4_swizzled_bf16("
          "Tensor merged_gate_up, Tensor! packed, Tensor! scales) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("silu_mul_quant_nvfp4_swizzled_bf16",
           torch::kCUDA,
           &silu_mul_quant_nvfp4_swizzled_bf16);
  ops.impl("silu_mul_merged_quant_nvfp4_swizzled_bf16",
           torch::kCUDA,
           &silu_mul_merged_quant_nvfp4_swizzled_bf16);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
