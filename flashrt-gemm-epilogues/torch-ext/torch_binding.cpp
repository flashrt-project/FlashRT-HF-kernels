#include <torch/all.h>
#include <torch/library.h>

#include <limits>

#if defined(CUDA_KERNEL)
#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAStream.h>
#define FLASHRT_DEVICE_GUARD(tensor) c10::OptionalDeviceGuard device_guard((tensor).device())
#define FLASHRT_CURRENT_STREAM(tensor) c10::cuda::getCurrentCUDAStream((tensor).get_device()).stream()
#elif defined(ROCM_KERNEL)
#include <c10/core/DeviceGuard.h>
#include <c10/hip/HIPStream.h>
#define FLASHRT_DEVICE_GUARD(tensor) c10::OptionalDeviceGuard device_guard((tensor).device())
#define FLASHRT_CURRENT_STREAM(tensor) c10::hip::getCurrentHIPStream((tensor).get_device()).stream()
#endif

#if defined(CUDA_KERNEL)
#include "bf16_gemm_bias_gelu.cuh"
#include "bias_gelu_quantize_fp8.cuh"
#include "channel_scale_quantize_fp8.cuh"
#elif defined(ROCM_KERNEL)
#include "rocm/gemm_epilogues_rocm.h"
#endif
#include "registration.h"
#include "torch_binding.h"

namespace {

void check_common_tensors(
    torch::Tensor const& input,
    torch::Tensor const& scale,
    torch::Tensor const& out) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(scale.is_cuda(), "scale must be a CUDA tensor");
  TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16,
              "input must have dtype torch.bfloat16");
  TORCH_CHECK(scale.scalar_type() == torch::kFloat32,
              "scale must have dtype torch.float32");
  #if defined(ROCM_KERNEL)
  TORCH_CHECK(out.scalar_type() == c10::ScalarType::Float8_e4m3fnuz,
              "out must have dtype torch.float8_e4m3fnuz on ROCm");
#else
  TORCH_CHECK(out.scalar_type() == c10::ScalarType::Float8_e4m3fn,
              "out must have dtype torch.float8_e4m3fn");
#endif
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(scale.is_contiguous(), "scale must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(input.sizes() == out.sizes(),
              "out must have the same shape as input");
  TORCH_CHECK(input.dim() >= 1, "input must have at least one dimension");
  TORCH_CHECK(input.numel() > 0, "input must be non-empty");
  TORCH_CHECK(scale.numel() == 1, "scale must contain exactly one value");
  TORCH_CHECK(input.get_device() == scale.get_device(),
              "input and scale must be on the same CUDA device");
  TORCH_CHECK(input.get_device() == out.get_device(),
              "input and out must be on the same CUDA device");
}

void launch_bias_gelu_quantize(
    torch::Tensor const& input,
    const void* bias_ptr,
    torch::Tensor const& scale,
    torch::Tensor& out) {
  const auto N64 = input.size(input.dim() - 1);
  TORCH_CHECK(N64 > 0, "last input dimension must be non-zero");
  TORCH_CHECK(N64 <= std::numeric_limits<int>::max(),
              "last input dimension must fit in int");
  const int N = static_cast<int>(N64);
  const long long M = input.numel() / N;

#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
  FLASHRT_DEVICE_GUARD(input);
  auto stream = FLASHRT_CURRENT_STREAM(input);
  flash_rt::quantize::bias_gelu_quantize_fp8_static_bf16(
      input.data_ptr(),
      bias_ptr,
      out.data_ptr(),
      reinterpret_cast<const float*>(scale.data_ptr()),
      M,
      N,
      stream);
#else
  TORCH_CHECK(false, "flashrt-gemm-epilogues was not built with CUDA/ROCm support");
#endif
}

int checked_last_dim(torch::Tensor const& input) {
  const auto K64 = input.size(input.dim() - 1);
  TORCH_CHECK(K64 > 0, "last input dimension must be non-zero");
  TORCH_CHECK(K64 <= std::numeric_limits<int>::max(),
              "last input dimension must fit in int");
  return static_cast<int>(K64);
}

void check_bf16_gemm_bias_gelu_tensors(
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& bias,
    torch::Tensor const& out) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
  TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
  TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
  TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
  TORCH_CHECK(a.scalar_type() == torch::kBFloat16,
              "a must have dtype torch.bfloat16");
  TORCH_CHECK(b.scalar_type() == torch::kBFloat16,
              "b must have dtype torch.bfloat16");
  TORCH_CHECK(bias.scalar_type() == torch::kBFloat16,
              "bias must have dtype torch.bfloat16");
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16,
              "out must have dtype torch.bfloat16");
  TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
  TORCH_CHECK(b.is_contiguous(), "b must be contiguous");
  TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(a.dim() == 2, "a must be a 2D tensor");
  TORCH_CHECK(b.dim() == 2, "b must be a 2D tensor");
  TORCH_CHECK(bias.dim() == 1, "bias must be a 1D tensor");
  TORCH_CHECK(out.dim() == 2, "out must be a 2D tensor");
  TORCH_CHECK(a.size(0) > 0 && a.size(1) > 0,
              "a dimensions must be non-zero");
  TORCH_CHECK(b.size(0) > 0 && b.size(1) > 0,
              "b dimensions must be non-zero");
  TORCH_CHECK(a.size(1) == b.size(0),
              "a.shape[1] must match b.shape[0]");
  TORCH_CHECK(out.size(0) == a.size(0) && out.size(1) == b.size(1),
              "out must have shape (a.shape[0], b.shape[1])");
  TORCH_CHECK(bias.size(0) == b.size(1),
              "bias length must match b.shape[1]");
  TORCH_CHECK(a.size(0) <= std::numeric_limits<int>::max(),
              "a.shape[0] must fit in int");
  TORCH_CHECK(b.size(1) <= std::numeric_limits<int>::max(),
              "b.shape[1] must fit in int");
  TORCH_CHECK(a.size(1) <= std::numeric_limits<int>::max(),
              "a.shape[1] must fit in int");
  TORCH_CHECK(a.get_device() == b.get_device(),
              "a and b must be on the same CUDA device");
  TORCH_CHECK(a.get_device() == bias.get_device(),
              "a and bias must be on the same CUDA device");
  TORCH_CHECK(a.get_device() == out.get_device(),
              "a and out must be on the same CUDA device");
}

void check_bf16_linear_tensors(
    torch::Tensor const& x,
    torch::Tensor const& w,
    torch::Tensor const& out) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(w.is_cuda(), "w must be a CUDA tensor");
  TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16,
              "x must have dtype torch.bfloat16");
  TORCH_CHECK(w.scalar_type() == torch::kBFloat16,
              "w must have dtype torch.bfloat16");
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16,
              "out must have dtype torch.bfloat16");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(w.is_contiguous(), "w must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(x.dim() == 2, "x must be a 2D tensor");
  TORCH_CHECK(w.dim() == 2, "w must be a 2D tensor");
  TORCH_CHECK(out.dim() == 2, "out must be a 2D tensor");
  TORCH_CHECK(x.size(0) > 0 && x.size(1) > 0,
              "x dimensions must be non-zero");
  TORCH_CHECK(w.size(0) > 0 && w.size(1) > 0,
              "w dimensions must be non-zero");
  TORCH_CHECK(x.size(1) == w.size(0),
              "x.shape[1] must match w.shape[0]");
  TORCH_CHECK(out.size(0) == x.size(0) && out.size(1) == w.size(1),
              "out must have shape (x.shape[0], w.shape[1])");
  TORCH_CHECK(x.size(0) <= std::numeric_limits<int>::max(),
              "x.shape[0] must fit in int");
  TORCH_CHECK(w.size(1) <= std::numeric_limits<int>::max(),
              "w.shape[1] must fit in int");
  TORCH_CHECK(x.size(1) <= std::numeric_limits<int>::max(),
              "x.shape[1] must fit in int");
  TORCH_CHECK(x.get_device() == w.get_device(),
              "x and w must be on the same CUDA device");
  TORCH_CHECK(x.get_device() == out.get_device(),
              "x and out must be on the same CUDA device");
}

void check_bf16_linear_bias_tensors(
    torch::Tensor const& x,
    torch::Tensor const& w,
    torch::Tensor const& bias,
    torch::Tensor const& out) {
  check_bf16_linear_tensors(x, w, out);
  TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
  TORCH_CHECK(bias.scalar_type() == torch::kBFloat16,
              "bias must have dtype torch.bfloat16");
  TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
  TORCH_CHECK(bias.dim() == 1, "bias must be a 1D tensor");
  TORCH_CHECK(bias.size(0) == w.size(1),
              "bias length must match w.shape[1]");
  TORCH_CHECK(x.get_device() == bias.get_device(),
              "x and bias must be on the same CUDA device");
}

}  // namespace

void bf16_linear_bf16(
    torch::Tensor const& x,
    torch::Tensor const& w,
    torch::Tensor& out) {
  check_bf16_linear_tensors(x, w, out);
  const int M = static_cast<int>(x.size(0));
  const int K = static_cast<int>(x.size(1));
  const int N = static_cast<int>(w.size(1));

#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
  FLASHRT_DEVICE_GUARD(x);
  auto stream = FLASHRT_CURRENT_STREAM(x);
  flash_rt::gemm::bf16_gemm(
      x.data_ptr(),
      w.data_ptr(),
      out.data_ptr(),
      M,
      N,
      K,
      stream);
#else
  TORCH_CHECK(false, "flashrt-gemm-epilogues was not built with CUDA/ROCm support");
#endif
}

void bf16_linear_bias_bf16(
    torch::Tensor const& x,
    torch::Tensor const& w,
    torch::Tensor const& bias,
    torch::Tensor& out) {
  check_bf16_linear_bias_tensors(x, w, bias, out);
  const int M = static_cast<int>(x.size(0));
  const int K = static_cast<int>(x.size(1));
  const int N = static_cast<int>(w.size(1));

#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
  FLASHRT_DEVICE_GUARD(x);
  auto stream = FLASHRT_CURRENT_STREAM(x);
  flash_rt::gemm::bf16_gemm_bias(
      x.data_ptr(),
      w.data_ptr(),
      bias.data_ptr(),
      out.data_ptr(),
      M,
      N,
      K,
      stream);
#else
  TORCH_CHECK(false, "flashrt-gemm-epilogues was not built with CUDA/ROCm support");
#endif
}

void bf16_gemm_bias_gelu(
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& bias,
    torch::Tensor& out) {
  check_bf16_gemm_bias_gelu_tensors(a, b, bias, out);
  const int M = static_cast<int>(a.size(0));
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(b.size(1));

#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
  FLASHRT_DEVICE_GUARD(a);
  auto stream = FLASHRT_CURRENT_STREAM(a);
  flash_rt::gemm::bf16_gemm_bias_gelu(
      a.data_ptr(),
      b.data_ptr(),
      bias.data_ptr(),
      out.data_ptr(),
      M,
      N,
      K,
      stream);
#else
  TORCH_CHECK(false, "flashrt-gemm-epilogues was not built with CUDA/ROCm support");
#endif
}

void bf16_gemm_bias(
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& bias,
    torch::Tensor& out) {
  check_bf16_gemm_bias_gelu_tensors(a, b, bias, out);
  const int M = static_cast<int>(a.size(0));
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(b.size(1));

#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
  FLASHRT_DEVICE_GUARD(a);
  auto stream = FLASHRT_CURRENT_STREAM(a);
  flash_rt::gemm::bf16_gemm_bias(
      a.data_ptr(),
      b.data_ptr(),
      bias.data_ptr(),
      out.data_ptr(),
      M,
      N,
      K,
      stream);
#else
  TORCH_CHECK(false, "flashrt-gemm-epilogues was not built with CUDA/ROCm support");
#endif
}

void bias_gelu_quantize_fp8_static_bf16(
    torch::Tensor const& input,
    torch::Tensor const& bias,
    torch::Tensor const& scale,
    torch::Tensor& out) {
  check_common_tensors(input, scale, out);
  TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
  TORCH_CHECK(bias.scalar_type() == torch::kBFloat16,
              "bias must have dtype torch.bfloat16");
  TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
  TORCH_CHECK(bias.dim() == 1, "bias must be a 1D tensor");
  TORCH_CHECK(bias.size(0) == input.size(input.dim() - 1),
              "bias length must match the last input dimension");
  TORCH_CHECK(input.get_device() == bias.get_device(),
              "input and bias must be on the same CUDA device");

  launch_bias_gelu_quantize(input, bias.data_ptr(), scale, out);
}

void gelu_quantize_fp8_static_bf16(
    torch::Tensor const& input,
    torch::Tensor const& scale,
    torch::Tensor& out) {
  check_common_tensors(input, scale, out);
  launch_bias_gelu_quantize(input, nullptr, scale, out);
}

void channel_scale_quantize_fp8_static_bf16(
    torch::Tensor const& input,
    torch::Tensor const& channel_scale,
    torch::Tensor const& scale,
    torch::Tensor& out) {
  check_common_tensors(input, scale, out);
  TORCH_CHECK(channel_scale.is_cuda(),
              "channel_scale must be a CUDA tensor");
  TORCH_CHECK(channel_scale.scalar_type() == torch::kBFloat16,
              "channel_scale must have dtype torch.bfloat16");
  TORCH_CHECK(channel_scale.is_contiguous(),
              "channel_scale must be contiguous");
  TORCH_CHECK(channel_scale.dim() == 1,
              "channel_scale must be a 1D tensor");
  const int K = checked_last_dim(input);
  TORCH_CHECK(channel_scale.size(0) == K,
              "channel_scale length must match the last input dimension");
  TORCH_CHECK(input.get_device() == channel_scale.get_device(),
              "input and channel_scale must be on the same CUDA device");
  const long long M = input.numel() / K;

#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
  FLASHRT_DEVICE_GUARD(input);
  auto stream = FLASHRT_CURRENT_STREAM(input);
  flash_rt::quantize::channel_scale_quantize_fp8_static_bf16(
      input.data_ptr(),
      channel_scale.data_ptr(),
      out.data_ptr(),
      reinterpret_cast<const float*>(scale.data_ptr()),
      M,
      K,
      stream);
#else
  TORCH_CHECK(false, "flashrt-gemm-epilogues was not built with CUDA/ROCm support");
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("bf16_linear_bf16("
          "Tensor x, Tensor w, Tensor! out) -> ()");
  ops.def("bf16_linear_bias_bf16("
          "Tensor x, Tensor w, Tensor bias, Tensor! out) -> ()");
  ops.def("bf16_gemm_bias_gelu("
          "Tensor a, Tensor b, Tensor bias, Tensor! out) -> ()");
  ops.def("bf16_gemm_bias("
          "Tensor a, Tensor b, Tensor bias, Tensor! out) -> ()");
  ops.def("bias_gelu_quantize_fp8_static_bf16("
          "Tensor input, Tensor bias, Tensor scale, Tensor! out) -> ()");
  ops.def("gelu_quantize_fp8_static_bf16("
          "Tensor input, Tensor scale, Tensor! out) -> ()");
  ops.def("channel_scale_quantize_fp8_static_bf16("
          "Tensor input, Tensor channel_scale, Tensor scale, Tensor! out) -> ()");
#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
  ops.impl("bf16_linear_bf16",
           torch::kCUDA,
           &bf16_linear_bf16);
  ops.impl("bf16_linear_bias_bf16",
           torch::kCUDA,
           &bf16_linear_bias_bf16);
  ops.impl("bf16_gemm_bias_gelu",
           torch::kCUDA,
           &bf16_gemm_bias_gelu);
  ops.impl("bf16_gemm_bias",
           torch::kCUDA,
           &bf16_gemm_bias);
  ops.impl("bias_gelu_quantize_fp8_static_bf16",
           torch::kCUDA,
           &bias_gelu_quantize_fp8_static_bf16);
  ops.impl("gelu_quantize_fp8_static_bf16",
           torch::kCUDA,
           &gelu_quantize_fp8_static_bf16);
  ops.impl("channel_scale_quantize_fp8_static_bf16",
           torch::kCUDA,
           &channel_scale_quantize_fp8_static_bf16);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
