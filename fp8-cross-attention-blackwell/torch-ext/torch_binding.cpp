// SPDX-License-Identifier: Apache-2.0
#include <torch/all.h>
#include <torch/library.h>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#endif

#include "registration.h"
#include "torch_binding.h"

#if defined(CUDA_KERNEL)
extern "C" int flashrt_fp8_gqa_cross_attention_sm100(
    const void*, const void*, const void*, void*, float*, void*, size_t,
    int, int, int, int, int, int, int, int,
    float, float, float, float, cudaStream_t);
#endif

namespace {
void check_fp8(torch::Tensor const& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous(),
              name, " must be contiguous CUDA");
  TORCH_CHECK(tensor.scalar_type() == c10::ScalarType::Float8_e4m3fn,
              name, " must have dtype torch.float8_e4m3fn");
  TORCH_CHECK(tensor.dim() == 4, name, " must have shape [B,S,H,128]");
}
}  // namespace

void fp8_gqa_cross_attention_bf16_out(
    torch::Tensor const& query,
    torch::Tensor const& key,
    torch::Tensor const& value,
    double query_scale,
    double key_scale,
    double value_scale,
    torch::Tensor& output,
    torch::Tensor& lse,
    torch::Tensor& workspace) {
  check_fp8(query, "query");
  check_fp8(key, "key");
  check_fp8(value, "value");
  TORCH_CHECK(key.sizes() == value.sizes(), "key and value shapes must match");
  TORCH_CHECK(query.size(0) == key.size(0), "query and key batch sizes must match");
  TORCH_CHECK(query.size(3) == 128 && key.size(3) == 128,
              "head dimension must be 128");
  TORCH_CHECK(query.size(2) > 0 && key.size(2) > 0 &&
                  query.size(2) % key.size(2) == 0,
              "query heads must be divisible by KV heads");
  TORCH_CHECK(output.is_cuda() && output.is_contiguous() &&
                  output.scalar_type() == torch::kBFloat16 &&
                  output.sizes() == query.sizes(),
              "output must be contiguous BF16 with the query shape");
  const int64_t rounded_sq = ((query.size(1) + 127) / 128) * 128;
  TORCH_CHECK(lse.is_cuda() && lse.is_contiguous() &&
                  lse.scalar_type() == torch::kFloat32 &&
                  lse.sizes() == torch::IntArrayRef(
                      {query.size(0), query.size(2), rounded_sq}),
              "lse must be contiguous FP32 [B,Hq,round_up(Sq,128)]");
  TORCH_CHECK(workspace.is_cuda() && workspace.is_contiguous() &&
                  workspace.scalar_type() == torch::kUInt8,
              "workspace must be a contiguous CUDA uint8 tensor");
  TORCH_CHECK(query.get_device() == key.get_device() &&
                  query.get_device() == value.get_device() &&
                  query.get_device() == output.get_device() &&
                  query.get_device() == lse.get_device() &&
                  query.get_device() == workspace.get_device(),
              "all tensors must be on the same CUDA device");
  TORCH_CHECK(query_scale > 0.0 && key_scale > 0.0 && value_scale > 0.0,
              "dequantization scales must be positive");

#if defined(CUDA_KERNEL)
  c10::cuda::CUDAGuard guard(query.device());
  auto* props = at::cuda::getDeviceProperties(query.get_device());
  const int capability = props->major * 10 + props->minor;
  TORCH_CHECK(capability == 100 || capability == 103 || capability == 110,
              "FP8 cross-attention requires SM100, SM103, or SM110");
  auto stream = at::cuda::getCurrentCUDAStream(query.get_device()).stream();
  int rc = flashrt_fp8_gqa_cross_attention_sm100(
      query.data_ptr(), key.data_ptr(), value.data_ptr(), output.data_ptr(),
      lse.data_ptr<float>(), workspace.data_ptr(), workspace.numel(),
      static_cast<int>(query.size(0)), static_cast<int>(query.size(1)),
      static_cast<int>(key.size(1)), static_cast<int>(query.size(2)),
      static_cast<int>(key.size(2)), 128,
      static_cast<int>(query.stride(1)), static_cast<int>(key.stride(1)),
      static_cast<float>(query_scale), static_cast<float>(key_scale),
      static_cast<float>(value_scale), 1.0f, stream);
  TORCH_CHECK(rc == 0, "FP8 GQA cross-attention failed with status ", rc);
#else
  TORCH_CHECK(false, "CUDA support was not built");
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("fp8_gqa_cross_attention_bf16_out(Tensor query, Tensor key, Tensor "
          "value, float query_scale, float key_scale, float value_scale, "
          "Tensor! output, Tensor! lse, Tensor! workspace) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("fp8_gqa_cross_attention_bf16_out", torch::kCUDA,
           &fp8_gqa_cross_attention_bf16_out);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
