#include "q_norm_rope_bf16.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdlib>

namespace flash_rt {
namespace vla_video {
namespace {

constexpr int kHeadDim = 128;
constexpr int kHalfDim = kHeadDim / 2;
constexpr int kThreads = kHeadDim;
constexpr int kWarps = kThreads / 32;

__device__ __forceinline__ float block_sum_4warp(float value, float* scratch) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffff, value, offset);
  }

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    scratch[warp] = value;
  }
  __syncthreads();

  if (warp == 0) {
    value = (lane < kWarps) ? scratch[lane] : 0.0f;
    #pragma unroll
    for (int offset = 2; offset > 0; offset >>= 1) {
      value += __shfl_xor_sync(0xffffffff, value, offset);
    }
    if (lane == 0) {
      scratch[0] = value;
    }
  }
  __syncthreads();
  return scratch[0];
}

__device__ __forceinline__ void write_norm_rope_row(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* cos,
    const __nv_bfloat16* sin,
    __nv_bfloat16* out,
    float eps) {
  const int tid = threadIdx.x;
  __shared__ float normed[kHeadDim];
  __shared__ float scratch[kWarps];

  const float value = __bfloat162float(input[tid]);
  const float square = value * value;
  const float sum_square = block_sum_4warp(square, scratch);
  const float rstd = rsqrtf(sum_square / static_cast<float>(kHeadDim) + eps);
  normed[tid] = value * rstd * __bfloat162float(weight[tid]);
  __syncthreads();

  if (tid < kHalfDim) {
    const float lo = normed[tid];
    const float hi = normed[tid + kHalfDim];
    const float c = __bfloat162float(cos[tid]);
    const float s = __bfloat162float(sin[tid]);
    out[tid] = __float2bfloat16(lo * c - hi * s);
  } else {
    const int idx = tid - kHalfDim;
    const float lo = normed[idx];
    const float hi = normed[tid];
    const float c = __bfloat162float(cos[idx]);
    const float s = __bfloat162float(sin[idx]);
    out[tid] = __float2bfloat16(hi * c + lo * s);
  }
}

__global__ void q_norm_rope_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    __nv_bfloat16* __restrict__ out,
    int rows,
    float eps) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  write_norm_rope_row(
      q + row * kHeadDim,
      weight,
      cos,
      sin,
      out + row * kHeadDim,
      eps);
}

__global__ void k_norm_rope_v_cache_kernel(
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out,
    int rows,
    float eps) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int offset = row * kHeadDim;
  write_norm_rope_row(
      k + offset,
      weight,
      cos,
      sin,
      k_out + offset,
      eps);
  v_out[offset + threadIdx.x] = v[offset + threadIdx.x];
}

__global__ void qkv_split_norm_rope_kernel(
    const __nv_bfloat16* __restrict__ packed_qkv,
    const __nv_bfloat16* __restrict__ norm_q_weight,
    const __nv_bfloat16* __restrict__ norm_k_weight,
    const float* __restrict__ freqs_re,
    const float* __restrict__ freqs_im,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    int tokens,
    int heads,
    int head_dim,
    int seq_len,
    float eps) {
  const int row = blockIdx.x;
  const int token = row % tokens;
  const int dim = heads * head_dim;
  const int dim2 = dim >> 1;
  const int head_dim2 = head_dim >> 1;
  const long long row_offset = static_cast<long long>(row) * 3 * dim;
  const long long out_offset = static_cast<long long>(row) * dim;

  __shared__ float scratch[33];
  const __nv_bfloat162* qkv2 =
      reinterpret_cast<const __nv_bfloat162*>(packed_qkv + row_offset);

  float q_sum = 0.0f;
  float k_sum = 0.0f;
  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    const __nv_bfloat162 qv = qkv2[i];
    const __nv_bfloat162 kv = qkv2[dim2 + i];
    const float q0 = __bfloat162float(qv.x);
    const float q1 = __bfloat162float(qv.y);
    const float k0 = __bfloat162float(kv.x);
    const float k1 = __bfloat162float(kv.y);
    q_sum += q0 * q0 + q1 * q1;
    k_sum += k0 * k0 + k1 * k1;
  }

  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    q_sum += __shfl_xor_sync(0xffffffff, q_sum, offset);
    k_sum += __shfl_xor_sync(0xffffffff, k_sum, offset);
  }
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    scratch[warp] = q_sum;
    scratch[16 + warp] = k_sum;
  }
  __syncthreads();
  if (warp == 0) {
    q_sum = (lane < (blockDim.x >> 5)) ? scratch[lane] : 0.0f;
    k_sum = (lane < (blockDim.x >> 5)) ? scratch[16 + lane] : 0.0f;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      q_sum += __shfl_xor_sync(0xffffffff, q_sum, offset);
      k_sum += __shfl_xor_sync(0xffffffff, k_sum, offset);
    }
    if (lane == 0) {
      scratch[0] = q_sum;
      scratch[1] = k_sum;
    }
  }
  __syncthreads();

  const float q_rstd = rsqrtf(scratch[0] / static_cast<float>(dim) + eps);
  const float k_rstd = rsqrtf(scratch[1] / static_cast<float>(dim) + eps);
  const bool apply_rope = token < seq_len;

  for (int pair = threadIdx.x; pair < heads * head_dim2; pair += blockDim.x) {
    const int head = pair / head_dim2;
    const int pair_in_head = pair - head * head_dim2;
    const int col_re = head * head_dim + 2 * pair_in_head;
    const int col_im = col_re + 1;

    const float q_re = __bfloat162float(packed_qkv[row_offset + col_re]);
    const float q_im = __bfloat162float(packed_qkv[row_offset + col_im]);
    const float k_re = __bfloat162float(packed_qkv[row_offset + dim + col_re]);
    const float k_im = __bfloat162float(packed_qkv[row_offset + dim + col_im]);
    const float qw_re = __bfloat162float(norm_q_weight[col_re]);
    const float qw_im = __bfloat162float(norm_q_weight[col_im]);
    const float kw_re = __bfloat162float(norm_k_weight[col_re]);
    const float kw_im = __bfloat162float(norm_k_weight[col_im]);

    const float qn_re = q_re * q_rstd * qw_re;
    const float qn_im = q_im * q_rstd * qw_im;
    const float kn_re = k_re * k_rstd * kw_re;
    const float kn_im = k_im * k_rstd * kw_im;

    float q_out_re = qn_re;
    float q_out_im = qn_im;
    float k_out_re = kn_re;
    float k_out_im = kn_im;
    if (apply_rope) {
      const long long freq_offset =
          static_cast<long long>(token) * head_dim2 + pair_in_head;
      const float c = freqs_re[freq_offset];
      const float s = freqs_im[freq_offset];
      q_out_re = qn_re * c - qn_im * s;
      q_out_im = qn_re * s + qn_im * c;
      k_out_re = kn_re * c - kn_im * s;
      k_out_im = kn_re * s + kn_im * c;
    }

    q_out[out_offset + col_re] = __float2bfloat16(q_out_re);
    q_out[out_offset + col_im] = __float2bfloat16(q_out_im);
    k_out[out_offset + col_re] = __float2bfloat16(k_out_re);
    k_out[out_offset + col_im] = __float2bfloat16(k_out_im);
  }
}

int qkv_rope_block_size(int tokens) {
  const char* value = std::getenv("FLASHRT_QKV_ROPE_BLOCK_SIZE");
  if (value != nullptr) {
    const int block_size = std::atoi(value);
    if (block_size == 128 || block_size == 256 || block_size == 512) {
      return block_size;
    }
  }
  if (tokens <= 64) {
    return 512;
  }
  return 256;
}

}  // namespace

void q_norm_rope_bf16(
    const void* q,
    const void* weight,
    const void* cos,
    const void* sin,
    void* out,
    int rows,
    float eps,
    cudaStream_t stream) {
  q_norm_rope_kernel<<<rows, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q),
      reinterpret_cast<const __nv_bfloat16*>(weight),
      reinterpret_cast<const __nv_bfloat16*>(cos),
      reinterpret_cast<const __nv_bfloat16*>(sin),
      reinterpret_cast<__nv_bfloat16*>(out),
      rows,
      eps);
}

void k_norm_rope_v_cache_bf16(
    const void* k,
    const void* v,
    const void* weight,
    const void* cos,
    const void* sin,
    void* k_out,
    void* v_out,
    int rows,
    float eps,
    cudaStream_t stream) {
  k_norm_rope_v_cache_kernel<<<rows, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(k),
      reinterpret_cast<const __nv_bfloat16*>(v),
      reinterpret_cast<const __nv_bfloat16*>(weight),
      reinterpret_cast<const __nv_bfloat16*>(cos),
      reinterpret_cast<const __nv_bfloat16*>(sin),
      reinterpret_cast<__nv_bfloat16*>(k_out),
      reinterpret_cast<__nv_bfloat16*>(v_out),
      rows,
      eps);
}

void qkv_split_norm_rope_bf16(
    const void* packed_qkv,
    const void* norm_q_weight,
    const void* norm_k_weight,
    const void* freqs_re,
    const void* freqs_im,
    void* q_out,
    void* k_out,
    int batch,
    int tokens,
    int heads,
    int head_dim,
    int seq_len,
    float eps,
    cudaStream_t stream) {
  const int blocks = batch * tokens;
  qkv_split_norm_rope_kernel<<<blocks, qkv_rope_block_size(tokens), 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(packed_qkv),
      reinterpret_cast<const __nv_bfloat16*>(norm_q_weight),
      reinterpret_cast<const __nv_bfloat16*>(norm_k_weight),
      reinterpret_cast<const float*>(freqs_re),
      reinterpret_cast<const float*>(freqs_im),
      reinterpret_cast<__nv_bfloat16*>(q_out),
      reinterpret_cast<__nv_bfloat16*>(k_out),
      tokens,
      heads,
      head_dim,
      seq_len,
      eps);
}

}  // namespace vla_video
}  // namespace flash_rt
