// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#include <limits>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#endif

#include "registration.h"
#include "torch_binding.h"
#include "xqa_bf16_fp8kv.cuh"

namespace {

constexpr int64_t kPageSize = 128;

struct XqaShape {
  int64_t num_q_heads;
  int64_t num_kv_heads;
  int64_t head_dim;
};

bool supported_shape(XqaShape const& shape) {
  return (shape.num_q_heads == 24 && shape.num_kv_heads == 4 &&
          shape.head_dim == 256) ||
         (shape.num_q_heads == 32 && shape.num_kv_heads == 8 &&
          shape.head_dim == 128) ||
         (shape.num_q_heads == 16 && shape.num_kv_heads == 8 &&
          shape.head_dim == 128);
}

void check_cuda_contiguous(torch::Tensor const& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_bf16(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16,
              name, " must have dtype torch.bfloat16");
}

void check_fp8_e4m3(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == c10::ScalarType::Float8_e4m3fn,
              name, " must have dtype torch.float8_e4m3fn");
}

void check_int32(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kInt32,
              name, " must have dtype torch.int32");
}

int checked_int(int64_t value, const char* name) {
  TORCH_CHECK(value > 0 && value <= std::numeric_limits<int>::max(),
              name, " must fit in positive int");
  return static_cast<int>(value);
}

int64_t q_seq_len_from_shape(torch::Tensor const& q, XqaShape* shape) {
  if (q.dim() == 3) {
    shape->num_q_heads = q.size(1);
    shape->head_dim = q.size(2);
    return q.size(0);
  }
  if (q.dim() == 5) {
    TORCH_CHECK(q.size(0) == 1 && q.size(1) == 1,
                "rank-5 q must begin with shape (1, 1, ...)");
    shape->num_q_heads = q.size(3);
    shape->head_dim = q.size(4);
    return q.size(2);
  }
  TORCH_CHECK(false, "q must have rank 3 or rank 5");
}

void check_same_device(torch::Tensor const& a, torch::Tensor const& b,
                       const char* an, const char* bn) {
  TORCH_CHECK(a.get_device() == b.get_device(),
              an, " and ", bn, " must be on the same CUDA device");
}

}  // namespace

void xqa_bf16_fp8kv(
    torch::Tensor const& q,
    torch::Tensor const& k_cache,
    torch::Tensor const& v_cache,
    torch::Tensor const& page_table,
    torch::Tensor const& seq_lens,
    torch::Tensor const& mask,
    torch::Tensor& out,
    torch::Tensor& semaphores,
    torch::Tensor& scratch,
    int64_t max_seq_len,
    double q_scale,
    double kv_scale,
    bool enable_pdl,
    int64_t sm_count,
    int64_t k_stride_page,
    int64_t k_stride_token,
    int64_t k_stride_head) {
  check_bf16(q, "q");
  check_bf16(out, "out");
  TORCH_CHECK(out.sizes() == q.sizes(), "out must have the same shape as q");
  XqaShape shape{};
  const int64_t q_seq = q_seq_len_from_shape(q, &shape);
  TORCH_CHECK(q_seq > 0 && q_seq <= 32,
              "v1 XQA package supports 1 <= q_seq <= 32 speculative/decode rows");

  check_fp8_e4m3(k_cache, "k_cache");
  check_fp8_e4m3(v_cache, "v_cache");
  TORCH_CHECK(k_cache.dim() == 4 && v_cache.sizes() == k_cache.sizes(),
              "k_cache/v_cache must have shape (pages, 128, num_kv_heads, head_dim)");
  shape.num_kv_heads = k_cache.size(2);
  TORCH_CHECK(supported_shape(shape),
              "unsupported XQA shape: q_heads=", shape.num_q_heads,
              ", kv_heads=", shape.num_kv_heads,
              ", head_dim=", shape.head_dim,
              "; supported configurations are 24/4/256, 32/8/128, and 16/8/128");
  TORCH_CHECK(k_cache.size(1) == kPageSize &&
                  k_cache.size(3) == shape.head_dim,
              "k_cache/v_cache page or head dimension does not match q");
  TORCH_CHECK(k_cache.size(0) > 0, "k_cache must contain at least one page");

  check_int32(page_table, "page_table");
  TORCH_CHECK(page_table.numel() >= k_cache.size(0),
              "page_table must contain at least one entry per cache page");
  check_int32(seq_lens, "seq_lens");
  TORCH_CHECK(seq_lens.numel() >= 1, "seq_lens must contain one sequence length");
  check_int32(mask, "mask");
  TORCH_CHECK(mask.numel() >= q_seq * ((q_seq + 31) / 32),
              "mask must have at least q_seq * ceil(q_seq / 32) elements");
  check_int32(semaphores, "semaphores");
  TORCH_CHECK(semaphores.numel() >= 256, "semaphores must contain at least 256 int32 entries");
  check_cuda_contiguous(scratch, "scratch");
  TORCH_CHECK(scratch.scalar_type() == torch::kUInt8, "scratch must have dtype torch.uint8");
  TORCH_CHECK(scratch.numel() >= (1 << 20), "scratch must contain at least 1 MiB");

  check_same_device(q, k_cache, "q", "k_cache");
  check_same_device(q, v_cache, "q", "v_cache");
  check_same_device(q, page_table, "q", "page_table");
  check_same_device(q, seq_lens, "q", "seq_lens");
  check_same_device(q, mask, "q", "mask");
  check_same_device(q, out, "q", "out");
  check_same_device(q, semaphores, "q", "semaphores");
  check_same_device(q, scratch, "q", "scratch");

  if (max_seq_len <= 0) {
    max_seq_len = k_cache.size(0) * kPageSize;
  }
  TORCH_CHECK(max_seq_len > 0 && max_seq_len <= k_cache.size(0) * kPageSize,
              "max_seq_len must be positive and covered by k_cache pages");
  TORCH_CHECK(max_seq_len % kPageSize == 0,
              "max_seq_len must be rounded to the 128-token page size");
  if (k_stride_page <= 0) {
    k_stride_page = kPageSize * shape.num_kv_heads * shape.head_dim;
  }
  if (k_stride_token <= 0) {
    k_stride_token = shape.num_kv_heads * shape.head_dim;
  }
  if (k_stride_head <= 0) k_stride_head = shape.head_dim;

#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(q.device());
  if (sm_count <= 0) {
    cudaDeviceProp prop{};
    const int device = q.get_device();
    TORCH_CHECK(cudaGetDeviceProperties(&prop, device) == cudaSuccess,
                "cudaGetDeviceProperties failed");
    sm_count = prop.multiProcessorCount;
  }
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device()).stream();
  if (shape.head_dim == 256) {
    flashrt_xqa_bf16_fp8kv(
        q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(),
        static_cast<const int32_t*>(page_table.data_ptr()),
        reinterpret_cast<const uint32_t*>(seq_lens.data_ptr()),
        reinterpret_cast<const uint32_t*>(mask.data_ptr()), out.data_ptr(),
        reinterpret_cast<uint32_t*>(semaphores.data_ptr()), scratch.data_ptr(),
        checked_int(max_seq_len, "max_seq_len"),
        checked_int(q_seq, "q_seq"), checked_int(sm_count, "sm_count"),
        static_cast<float>(q_scale), static_cast<float>(kv_scale), enable_pdl,
        k_stride_page, k_stride_token, k_stride_head, stream);
  } else {
    flashrt_xqa_bf16_fp8kv_d128(
        q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(),
        static_cast<const int32_t*>(page_table.data_ptr()),
        reinterpret_cast<const uint32_t*>(seq_lens.data_ptr()),
        reinterpret_cast<const uint32_t*>(mask.data_ptr()), out.data_ptr(),
        reinterpret_cast<uint32_t*>(semaphores.data_ptr()), scratch.data_ptr(),
        checked_int(max_seq_len, "max_seq_len"),
        checked_int(q_seq, "q_seq"),
        checked_int(shape.num_kv_heads, "num_kv_heads"),
        checked_int(shape.num_q_heads / shape.num_kv_heads, "head_group_size"),
        checked_int(sm_count, "sm_count"), static_cast<float>(q_scale),
        static_cast<float>(kv_scale), enable_pdl, k_stride_page, k_stride_token,
        k_stride_head, stream);
  }
#else
  TORCH_CHECK(false, "fp8-kv-attention was not built with CUDA support");
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("xqa_bf16_fp8kv(Tensor q, Tensor k_cache, Tensor v_cache, Tensor page_table, Tensor seq_lens, Tensor mask, Tensor! out, Tensor! semaphores, Tensor! scratch, int max_seq_len=0, float q_scale=1.0, float kv_scale=1.0, bool enable_pdl=True, int sm_count=0, int k_stride_page=0, int k_stride_token=0, int k_stride_head=0) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("xqa_bf16_fp8kv", torch::kCUDA, &xqa_bf16_fp8kv);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
