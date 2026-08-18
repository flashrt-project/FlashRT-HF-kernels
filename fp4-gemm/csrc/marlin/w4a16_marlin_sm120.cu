// Standalone BF16 x NVFP4 Marlin launcher for M <= 16.
//
// The device implementation is derived from vLLM's Apache-2.0 Marlin
// backend. This adapter deliberately accepts weights/scales already converted
// at bind time to Marlin layout, so the runtime hot path is one graph-safe
// kernel launch with caller-owned output and lock workspace.

#include "marlin/w4a16_marlin_sm120.cuh"

#include <cuda_runtime.h>

#include "marlin/core/scalar_type.hpp"

namespace marlin {
void marlin_mm(const void* A, const void* B, void* C, void* C_tmp,
               void* b_bias, void* a_s, void* b_s, void* g_s, void* zp,
               void* g_idx, void* perm, void* a_tmp, int prob_m, int prob_n,
               int prob_k, int lda, void* workspace,
               vllm::ScalarType const& a_type,
               vllm::ScalarType const& b_type,
               vllm::ScalarType const& c_type,
               vllm::ScalarType const& s_type, bool has_bias,
               bool has_act_order, bool is_k_full, bool has_zp,
               int num_groups, int group_size, int dev, cudaStream_t stream,
               int thread_k_init, int thread_n_init, int sms,
               bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float);
}  // namespace marlin

namespace flash_rt::gemm {

int w4a16_marlin_sm120_bf16(const void* a, const void* b_marlin,
                            const void* scales_marlin,
                            const void* global_scale, void* workspace,
                            void* out, int m, int n, int k, int lda,
                            cudaStream_t stream) {
  if (a == nullptr || b_marlin == nullptr || scales_marlin == nullptr ||
      global_scale == nullptr || workspace == nullptr || out == nullptr ||
      m < 1 || m > 16 || n <= 0 || n % 64 != 0 || k <= 0 || k % 128 != 0 ||
      lda < k) {
    return 1;
  }

  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) return 2;
  cudaDeviceProp props{};
  if (cudaGetDeviceProperties(&props, device) != cudaSuccess) return 3;
  if (props.major != 12 || props.minor != 0) return 4;

  marlin::marlin_mm(
      a, b_marlin, out, nullptr, nullptr, nullptr,
      const_cast<void*>(scales_marlin), const_cast<void*>(global_scale),
      nullptr, nullptr, nullptr, nullptr, m, n, k, lda, workspace,
      vllm::kBFloat16, vllm::kFE2M1f, vllm::kBFloat16, vllm::kFE4M3fn,
      false, false, true, false, k / 16, 16, device, stream, -1, -1,
      props.multiProcessorCount, false, false, false);
  return cudaGetLastError() == cudaSuccess ? 0 : 5;
}

}  // namespace flash_rt::gemm
