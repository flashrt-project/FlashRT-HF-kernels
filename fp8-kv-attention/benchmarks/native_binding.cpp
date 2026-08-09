#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "xqa_bf16_fp8kv.cuh"

namespace {

void xqa(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor page_table,
    torch::Tensor seq_lens,
    torch::Tensor mask,
    torch::Tensor out,
    torch::Tensor semaphores,
    torch::Tensor scratch,
    int64_t max_seq_len,
    double q_scale,
    double kv_scale,
    bool enable_pdl) {
  const int q_seq = q.dim() == 3 ? q.size(0) : q.size(2);
  const int q_heads = q.dim() == 3 ? q.size(1) : q.size(3);
  const int head_dim = q.dim() == 3 ? q.size(2) : q.size(4);
  const int kv_heads = k_cache.size(2);
  cudaDeviceProp prop{};
  TORCH_CHECK(cudaGetDeviceProperties(&prop, q.get_device()) == cudaSuccess);
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device()).stream();
  const int64_t stride_page = 128 * kv_heads * head_dim;
  const int64_t stride_token = kv_heads * head_dim;
  if (head_dim == 256) {
    if (q_heads == 16 && kv_heads == 2) {
      flashrt_xqa_bf16_fp8kv_d256_g8(
          q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(),
          static_cast<const int32_t*>(page_table.data_ptr()),
          reinterpret_cast<const uint32_t*>(seq_lens.data_ptr()),
          reinterpret_cast<const uint32_t*>(mask.data_ptr()), out.data_ptr(),
          reinterpret_cast<uint32_t*>(semaphores.data_ptr()),
          scratch.data_ptr(), max_seq_len, q_seq, kv_heads,
          prop.multiProcessorCount, static_cast<float>(q_scale),
          static_cast<float>(kv_scale), enable_pdl, stride_page, stride_token,
          head_dim, stream);
    } else {
      flashrt_xqa_bf16_fp8kv(
        q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(),
        static_cast<const int32_t*>(page_table.data_ptr()),
        reinterpret_cast<const uint32_t*>(seq_lens.data_ptr()),
        reinterpret_cast<const uint32_t*>(mask.data_ptr()), out.data_ptr(),
        reinterpret_cast<uint32_t*>(semaphores.data_ptr()),
        scratch.data_ptr(), max_seq_len, q_seq, prop.multiProcessorCount,
        static_cast<float>(q_scale), static_cast<float>(kv_scale), enable_pdl,
        stride_page, stride_token, head_dim, stream);
    }
  } else {
    flashrt_xqa_bf16_fp8kv_d128(
        q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(),
        static_cast<const int32_t*>(page_table.data_ptr()),
        reinterpret_cast<const uint32_t*>(seq_lens.data_ptr()),
        reinterpret_cast<const uint32_t*>(mask.data_ptr()), out.data_ptr(),
        reinterpret_cast<uint32_t*>(semaphores.data_ptr()),
        scratch.data_ptr(), max_seq_len, q_seq, kv_heads,
        q_heads / kv_heads, prop.multiProcessorCount,
        static_cast<float>(q_scale), static_cast<float>(kv_scale), enable_pdl,
        stride_page, stride_token, head_dim, stream);
  }
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("xqa", &xqa);
}
