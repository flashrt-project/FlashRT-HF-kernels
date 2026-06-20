// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#include <limits>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "fused_fp4/norm_silu_fp4_sfa.cuh"
#include "fused_fp4/dequantize_fp4_sfa.cuh"
#include "fused_fp4/silu_mul_two_fp4_to_fp4.cuh"
#include "quantize/reshape_scales_sfa.cuh"
#include "registration.h"
#include "torch_binding.h"

namespace {

void check_cuda_contiguous(torch::Tensor const& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_fp16_matrix(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16,
              name, " must have dtype torch.float16");
  TORCH_CHECK(tensor.dim() == 2, name, " must have shape (rows, cols)");
  TORCH_CHECK(tensor.size(0) > 0 && tensor.size(1) > 0,
              name, " dimensions must be positive");
  TORCH_CHECK(tensor.size(1) % 16 == 0, name, ".shape[1] must be divisible by 16");
}

void check_uint8(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kUInt8,
              name, " must have dtype torch.uint8");
}

void check_same_device(torch::Tensor const& a, torch::Tensor const& b,
                       const char* a_name, const char* b_name) {
  TORCH_CHECK(a.get_device() == b.get_device(),
              a_name, " and ", b_name, " must be on the same CUDA device");
}

int checked_int(int64_t value, const char* name) {
  TORCH_CHECK(value > 0 && value <= std::numeric_limits<int>::max(),
              name, " must fit in positive int");
  return static_cast<int>(value);
}

void check_packed_sfa(
    torch::Tensor const& packed,
    torch::Tensor const& sfa,
    torch::Tensor const& input,
    int64_t rows,
    int64_t dim) {
  check_uint8(packed, "packed");
  check_uint8(sfa, "sfa");
  TORCH_CHECK(packed.sizes() == torch::IntArrayRef({rows, dim / 2}),
              "packed must have shape (rows, dim / 2)");
  TORCH_CHECK(sfa.dim() == 1, "sfa must be a one-dimensional byte buffer");
  const int64_t required = sfa_size_bytes(rows, dim, false);
  TORCH_CHECK(required > 0, "sfa_size_bytes returned invalid size");
  TORCH_CHECK(sfa.numel() >= required,
              "sfa is too small for CUTLASS SFA layout");
  check_same_device(input, packed, "input", "packed");
  check_same_device(input, sfa, "input", "sfa");
}

struct TwoFp4Shape {
  int64_t rows;
  int64_t hidden;
};

TwoFp4Shape check_two_fp4_inputs(
    torch::Tensor const& gate_packed,
    torch::Tensor const& gate_sfa,
    torch::Tensor const& up_packed,
    torch::Tensor const& up_sfa,
    torch::Tensor const& out_packed,
    torch::Tensor const& out_sfa) {
  check_uint8(gate_packed, "gate_packed");
  check_uint8(gate_sfa, "gate_sfa");
  check_uint8(up_packed, "up_packed");
  check_uint8(up_sfa, "up_sfa");
  check_uint8(out_packed, "out_packed");
  check_uint8(out_sfa, "out_sfa");
  TORCH_CHECK(gate_packed.dim() == 2, "gate_packed must have shape (rows, hidden / 2)");
  TORCH_CHECK(gate_packed.size(0) > 0 && gate_packed.size(1) > 0,
              "gate_packed dimensions must be positive");
  TORCH_CHECK(up_packed.sizes() == gate_packed.sizes() &&
              out_packed.sizes() == gate_packed.sizes(),
              "up_packed and out_packed must match gate_packed shape");
  const int64_t rows = gate_packed.size(0);
  const int64_t hidden = gate_packed.size(1) * 2;
  const int64_t required = sfa_size_bytes(rows, hidden, false);
  TORCH_CHECK(gate_sfa.numel() >= required && up_sfa.numel() >= required &&
              out_sfa.numel() >= required,
              "SFA buffers are too small");
  check_same_device(gate_packed, gate_sfa, "gate_packed", "gate_sfa");
  check_same_device(gate_packed, up_packed, "gate_packed", "up_packed");
  check_same_device(gate_packed, up_sfa, "gate_packed", "up_sfa");
  check_same_device(gate_packed, out_packed, "gate_packed", "out_packed");
  check_same_device(gate_packed, out_sfa, "gate_packed", "out_sfa");
  return {rows, hidden};
}

}  // namespace

int64_t sfa_size_bytes(int64_t rows, int64_t dim, bool is_sfb) {
  TORCH_CHECK(rows > 0 && dim > 0, "rows and dim must be positive");
  TORCH_CHECK(dim % 16 == 0, "dim must be divisible by 16");
#if defined(CUDA_KERNEL)
  return flash_rt::fp4::sfa_size_bytes(
      checked_int(rows, "rows"), checked_int(dim, "dim"), is_sfb);
#else
  TORCH_CHECK(false, "fp4-fused-ops was not built with CUDA support");
#endif
}

int64_t sfa_size_bytes_for(
    torch::Tensor const& anchor,
    int64_t rows,
    int64_t dim,
    bool is_sfb) {
  check_cuda_contiguous(anchor, "anchor");
  return sfa_size_bytes(rows, dim, is_sfb);
}

void rms_norm_fp4_sfa_fp16(
    torch::Tensor const& x,
    torch::Tensor& packed,
    torch::Tensor& sfa) {
  check_fp16_matrix(x, "x");
  TORCH_CHECK(x.size(1) <= 2048,
              "rms_norm_fp4_sfa_fp16 supports dim <= 2048; use a v2 producer for larger dim");
  const int64_t rows = x.size(0);
  const int64_t dim = x.size(1);
  check_packed_sfa(packed, sfa, x, rows, dim);
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::fused_fp4::rms_norm_fp4_sfa_fp16(
      reinterpret_cast<const __half*>(x.data_ptr()),
      reinterpret_cast<uint8_t*>(packed.data_ptr()),
      reinterpret_cast<uint8_t*>(sfa.data_ptr()),
      checked_int(rows, "rows"),
      checked_int(dim, "dim"),
      stream);
#endif
}

void residual_add_rms_norm_fp4_sfa_fp16(
    torch::Tensor& residual,
    torch::Tensor const& x,
    torch::Tensor& packed,
    torch::Tensor& sfa) {
  check_fp16_matrix(residual, "residual");
  check_fp16_matrix(x, "x");
  TORCH_CHECK(residual.sizes() == x.sizes(), "residual and x must have the same shape");
  TORCH_CHECK(x.size(1) <= 2048,
              "residual_add_rms_norm_fp4_sfa_fp16 supports dim <= 2048; use residual_add_rms_norm_fp4_sfa_v2_fp16 for larger dim");
  check_same_device(residual, x, "residual", "x");
  const int64_t rows = x.size(0);
  const int64_t dim = x.size(1);
  check_packed_sfa(packed, sfa, x, rows, dim);
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::fused_fp4::residual_add_rms_norm_fp4_sfa_fp16(
      reinterpret_cast<__half*>(residual.data_ptr()),
      reinterpret_cast<const __half*>(x.data_ptr()),
      reinterpret_cast<uint8_t*>(packed.data_ptr()),
      reinterpret_cast<uint8_t*>(sfa.data_ptr()),
      checked_int(rows, "rows"),
      checked_int(dim, "dim"),
      stream);
#endif
}

void residual_add_rms_norm_fp4_sfa_v2_fp16(
    torch::Tensor& residual,
    torch::Tensor const& x,
    torch::Tensor& packed,
    torch::Tensor& sfa) {
  check_fp16_matrix(residual, "residual");
  check_fp16_matrix(x, "x");
  TORCH_CHECK(residual.sizes() == x.sizes(), "residual and x must have the same shape");
  TORCH_CHECK(x.size(1) <= 16384,
              "v2 residual path supports dim <= 16384");
  check_same_device(residual, x, "residual", "x");
  const int64_t rows = x.size(0);
  const int64_t dim = x.size(1);
  check_packed_sfa(packed, sfa, x, rows, dim);
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::fused_fp4::residual_add_rms_norm_fp4_sfa_v2_fp16(
      reinterpret_cast<__half*>(residual.data_ptr()),
      reinterpret_cast<const __half*>(x.data_ptr()),
      reinterpret_cast<uint8_t*>(packed.data_ptr()),
      reinterpret_cast<uint8_t*>(sfa.data_ptr()),
      checked_int(rows, "rows"),
      checked_int(dim, "dim"),
      stream);
#endif
}

void residual_add_rms_norm_mul_fp4_sfa_fp16(
    torch::Tensor& residual,
    torch::Tensor const& x,
    torch::Tensor const& inv_s,
    torch::Tensor& packed,
    torch::Tensor& sfa) {
  check_fp16_matrix(residual, "residual");
  check_fp16_matrix(x, "x");
  check_cuda_contiguous(inv_s, "inv_s");
  TORCH_CHECK(inv_s.scalar_type() == torch::kFloat16,
              "inv_s must have dtype torch.float16");
  TORCH_CHECK(residual.sizes() == x.sizes(), "residual and x must have the same shape");
  TORCH_CHECK(x.size(1) <= 2048,
              "residual_add_rms_norm_mul_fp4_sfa_fp16 supports dim <= 2048");
  TORCH_CHECK(inv_s.sizes() == torch::IntArrayRef({x.size(1)}),
              "inv_s must have shape (dim,)");
  check_same_device(residual, x, "residual", "x");
  check_same_device(residual, inv_s, "residual", "inv_s");
  const int64_t rows = x.size(0);
  const int64_t dim = x.size(1);
  check_packed_sfa(packed, sfa, x, rows, dim);
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  flash_rt::fused_fp4::residual_add_rms_norm_mul_fp4_sfa_fp16(
      reinterpret_cast<__half*>(residual.data_ptr()),
      reinterpret_cast<const __half*>(x.data_ptr()),
      reinterpret_cast<const __half*>(inv_s.data_ptr()),
      reinterpret_cast<uint8_t*>(packed.data_ptr()),
      reinterpret_cast<uint8_t*>(sfa.data_ptr()),
      checked_int(rows, "rows"),
      checked_int(dim, "dim"),
      stream);
#endif
}

void silu_mul_fp4_sfa_fp16(
    torch::Tensor const& merged,
    torch::Tensor& packed,
    torch::Tensor& sfa) {
  check_fp16_matrix(merged, "merged");
  TORCH_CHECK(merged.size(1) % 32 == 0,
              "merged.shape[1] must be divisible by 32 because hidden dim is half");
  const int64_t rows = merged.size(0);
  const int64_t hidden = merged.size(1) / 2;
  check_packed_sfa(packed, sfa, merged, rows, hidden);
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(merged.device());
  auto stream = at::cuda::getCurrentCUDAStream(merged.get_device()).stream();
  flash_rt::fused_fp4::gate_silu_mul_fp4_sfa_fp16(
      reinterpret_cast<const __half*>(merged.data_ptr()),
      reinterpret_cast<uint8_t*>(packed.data_ptr()),
      reinterpret_cast<uint8_t*>(sfa.data_ptr()),
      checked_int(rows, "rows"),
      checked_int(hidden, "hidden"),
      stream);
#endif
}

void silu_mul_fp4_sfa_v2_fp16(
    torch::Tensor const& merged,
    torch::Tensor& packed,
    torch::Tensor& sfa) {
  check_fp16_matrix(merged, "merged");
  TORCH_CHECK(merged.size(1) % 32 == 0,
              "merged.shape[1] must be divisible by 32 because hidden dim is half");
  const int64_t rows = merged.size(0);
  const int64_t hidden = merged.size(1) / 2;
  check_packed_sfa(packed, sfa, merged, rows, hidden);
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(merged.device());
  auto stream = at::cuda::getCurrentCUDAStream(merged.get_device()).stream();
  flash_rt::fused_fp4::gate_silu_mul_fp4_sfa_v2_fp16(
      reinterpret_cast<const __half*>(merged.data_ptr()),
      reinterpret_cast<uint8_t*>(packed.data_ptr()),
      reinterpret_cast<uint8_t*>(sfa.data_ptr()),
      checked_int(rows, "rows"),
      checked_int(hidden, "hidden"),
      stream);
#endif
}

void silu_mul_mul_fp4_sfa_v2_fp16(
    torch::Tensor const& merged,
    torch::Tensor const& inv_s,
    torch::Tensor& packed,
    torch::Tensor& sfa) {
  check_fp16_matrix(merged, "merged");
  check_cuda_contiguous(inv_s, "inv_s");
  TORCH_CHECK(inv_s.scalar_type() == torch::kFloat16,
              "inv_s must have dtype torch.float16");
  TORCH_CHECK(merged.size(1) % 32 == 0,
              "merged.shape[1] must be divisible by 32 because hidden dim is half");
  const int64_t rows = merged.size(0);
  const int64_t hidden = merged.size(1) / 2;
  TORCH_CHECK(inv_s.sizes() == torch::IntArrayRef({hidden}),
              "inv_s must have shape (hidden,)");
  check_same_device(merged, inv_s, "merged", "inv_s");
  check_packed_sfa(packed, sfa, merged, rows, hidden);
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(merged.device());
  auto stream = at::cuda::getCurrentCUDAStream(merged.get_device()).stream();
  flash_rt::fused_fp4::gate_silu_mul_mul_fp4_sfa_v2_fp16(
      reinterpret_cast<const __half*>(merged.data_ptr()),
      reinterpret_cast<const __half*>(inv_s.data_ptr()),
      reinterpret_cast<uint8_t*>(packed.data_ptr()),
      reinterpret_cast<uint8_t*>(sfa.data_ptr()),
      checked_int(rows, "rows"),
      checked_int(hidden, "hidden"),
      stream);
#endif
}

void silu_mul_two_fp4_to_fp4(
    torch::Tensor const& gate_packed,
    torch::Tensor const& gate_sfa,
    torch::Tensor const& up_packed,
    torch::Tensor const& up_sfa,
    torch::Tensor& out_packed,
    torch::Tensor& out_sfa) {
  auto shape = check_two_fp4_inputs(
      gate_packed, gate_sfa, up_packed, up_sfa, out_packed, out_sfa);
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(gate_packed.device());
  auto stream = at::cuda::getCurrentCUDAStream(gate_packed.get_device()).stream();
  flash_rt::fused_fp4::silu_mul_two_fp4_to_fp4(
      reinterpret_cast<const uint8_t*>(gate_packed.data_ptr()),
      reinterpret_cast<const uint8_t*>(gate_sfa.data_ptr()),
      reinterpret_cast<const uint8_t*>(up_packed.data_ptr()),
      reinterpret_cast<const uint8_t*>(up_sfa.data_ptr()),
      reinterpret_cast<uint8_t*>(out_packed.data_ptr()),
      reinterpret_cast<uint8_t*>(out_sfa.data_ptr()),
      checked_int(shape.rows, "rows"),
      checked_int(shape.hidden, "hidden"),
      stream);
#endif
}

void silu_mul_two_mul_fp4_to_fp4(
    torch::Tensor const& gate_packed,
    torch::Tensor const& gate_sfa,
    torch::Tensor const& up_packed,
    torch::Tensor const& up_sfa,
    torch::Tensor const& inv_s,
    torch::Tensor& out_packed,
    torch::Tensor& out_sfa) {
  check_cuda_contiguous(inv_s, "inv_s");
  TORCH_CHECK(inv_s.scalar_type() == torch::kFloat16,
              "inv_s must have dtype torch.float16");
  auto shape = check_two_fp4_inputs(
      gate_packed, gate_sfa, up_packed, up_sfa, out_packed, out_sfa);
  TORCH_CHECK(inv_s.dim() == 1 && inv_s.size(0) == shape.hidden,
              "inv_s must have shape (hidden,)");
  check_same_device(gate_packed, inv_s, "gate_packed", "inv_s");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(gate_packed.device());
  auto stream = at::cuda::getCurrentCUDAStream(gate_packed.get_device()).stream();
  flash_rt::fused_fp4::silu_mul_two_mul_fp4_to_fp4(
      reinterpret_cast<const uint8_t*>(gate_packed.data_ptr()),
      reinterpret_cast<const uint8_t*>(gate_sfa.data_ptr()),
      reinterpret_cast<const uint8_t*>(up_packed.data_ptr()),
      reinterpret_cast<const uint8_t*>(up_sfa.data_ptr()),
      reinterpret_cast<const __half*>(inv_s.data_ptr()),
      reinterpret_cast<uint8_t*>(out_packed.data_ptr()),
      reinterpret_cast<uint8_t*>(out_sfa.data_ptr()),
      checked_int(shape.rows, "rows"),
      checked_int(shape.hidden, "hidden"),
      stream);
#endif
}

void dequantize_fp4_sfa_fp16(
    torch::Tensor const& packed,
    torch::Tensor const& sfa,
    torch::Tensor& out) {
  check_uint8(packed, "packed");
  check_uint8(sfa, "sfa");
  check_cuda_contiguous(out, "out");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16,
              "out must have dtype torch.float16");
  TORCH_CHECK(out.dim() == 2, "out must have shape (rows, dim)");
  const int64_t rows = out.size(0);
  const int64_t dim = out.size(1);
  TORCH_CHECK(dim % 16 == 0, "out.shape[1] must be divisible by 16");
  TORCH_CHECK(packed.sizes() == torch::IntArrayRef({rows, dim / 2}),
              "packed must have shape (rows, dim / 2)");
  const int64_t required = sfa_size_bytes(rows, dim, false);
  TORCH_CHECK(sfa.numel() >= required,
              "sfa is too small for CUTLASS SFA layout");
  check_same_device(packed, sfa, "packed", "sfa");
  check_same_device(packed, out, "packed", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(packed.device());
  auto stream = at::cuda::getCurrentCUDAStream(packed.get_device()).stream();
  flash_rt::fused_fp4::dequantize_fp4_sfa_fp16(
      reinterpret_cast<const uint8_t*>(packed.data_ptr()),
      reinterpret_cast<const uint8_t*>(sfa.data_ptr()),
      reinterpret_cast<__half*>(out.data_ptr()),
      checked_int(rows, "rows"),
      checked_int(dim, "dim"),
      stream);
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("sfa_size_bytes_for(Tensor anchor, int rows, int dim, bool is_sfb=False) -> int");
  ops.def("rms_norm_fp4_sfa_fp16(Tensor x, Tensor! packed, Tensor! sfa) -> ()");
  ops.def("residual_add_rms_norm_fp4_sfa_fp16(Tensor! residual, Tensor x, Tensor! packed, Tensor! sfa) -> ()");
  ops.def("residual_add_rms_norm_fp4_sfa_v2_fp16(Tensor! residual, Tensor x, Tensor! packed, Tensor! sfa) -> ()");
  ops.def("residual_add_rms_norm_mul_fp4_sfa_fp16(Tensor! residual, Tensor x, Tensor inv_s, Tensor! packed, Tensor! sfa) -> ()");
  ops.def("silu_mul_fp4_sfa_fp16(Tensor merged, Tensor! packed, Tensor! sfa) -> ()");
  ops.def("silu_mul_fp4_sfa_v2_fp16(Tensor merged, Tensor! packed, Tensor! sfa) -> ()");
  ops.def("silu_mul_mul_fp4_sfa_v2_fp16(Tensor merged, Tensor inv_s, Tensor! packed, Tensor! sfa) -> ()");
  ops.def("silu_mul_two_fp4_to_fp4(Tensor gate_packed, Tensor gate_sfa, Tensor up_packed, Tensor up_sfa, Tensor! out_packed, Tensor! out_sfa) -> ()");
  ops.def("silu_mul_two_mul_fp4_to_fp4(Tensor gate_packed, Tensor gate_sfa, Tensor up_packed, Tensor up_sfa, Tensor inv_s, Tensor! out_packed, Tensor! out_sfa) -> ()");
  ops.def("dequantize_fp4_sfa_fp16(Tensor packed, Tensor sfa, Tensor! out) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("sfa_size_bytes_for", torch::kCUDA, &sfa_size_bytes_for);
  ops.impl("rms_norm_fp4_sfa_fp16", torch::kCUDA, &rms_norm_fp4_sfa_fp16);
  ops.impl("residual_add_rms_norm_fp4_sfa_fp16", torch::kCUDA, &residual_add_rms_norm_fp4_sfa_fp16);
  ops.impl("residual_add_rms_norm_fp4_sfa_v2_fp16", torch::kCUDA, &residual_add_rms_norm_fp4_sfa_v2_fp16);
  ops.impl("residual_add_rms_norm_mul_fp4_sfa_fp16", torch::kCUDA, &residual_add_rms_norm_mul_fp4_sfa_fp16);
  ops.impl("silu_mul_fp4_sfa_fp16", torch::kCUDA, &silu_mul_fp4_sfa_fp16);
  ops.impl("silu_mul_fp4_sfa_v2_fp16", torch::kCUDA, &silu_mul_fp4_sfa_v2_fp16);
  ops.impl("silu_mul_mul_fp4_sfa_v2_fp16", torch::kCUDA, &silu_mul_mul_fp4_sfa_v2_fp16);
  ops.impl("silu_mul_two_fp4_to_fp4", torch::kCUDA, &silu_mul_two_fp4_to_fp4);
  ops.impl("silu_mul_two_mul_fp4_to_fp4", torch::kCUDA, &silu_mul_two_mul_fp4_to_fp4);
  ops.impl("dequantize_fp4_sfa_fp16", torch::kCUDA, &dequantize_fp4_sfa_fp16);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
