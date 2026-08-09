// SPDX-License-Identifier: Apache-2.0
//
// Gated DeltaNet recurrent (single-token decode) kernel.
//
// Block layout: one block per (b, h) where h indexes ``num_v_heads``.
// Within a block, threadIdx.x = t in [0, head_v_dim) owns column t of
// the state matrix state[b, h, :, t] (head_k_dim elements).
//
// Per-thread state column lives in registers (head_k_dim fp32 = 128
// regs/thread on Qwen3.6). Q/K/V are loaded into shared memory once
// per block, then broadcast across threads.

#include "gated_delta_attention.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

namespace {

constexpr int kHD = 128;   // Qwen3.6 head_k_dim == head_v_dim
constexpr int kQHeads = 16;
constexpr int kVHeads = 48;
constexpr int kWyChunk = 64;
constexpr float kEps = 1e-6f;
constexpr int kSplitThreads = 256;

template <int HD>
__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
  // Warp reduce.
  for (int off = 16; off > 0; off >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, off);
  }
  // Cross-warp reduce via smem (4 warps for HD=128).
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) smem[warp] = val;
  __syncthreads();

  if (warp == 0) {
    val = (lane < (HD / 32)) ? smem[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) {
      val += __shfl_xor_sync(0xffffffff, val, off);
    }
    if (lane == 0) smem[0] = val;
  }
  __syncthreads();
  return smem[0];
}

template <int HD>
__global__ void gated_deltanet_recurrent_kernel(
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ v_in,
    const __nv_bfloat16* __restrict__ g_in,
    const __nv_bfloat16* __restrict__ beta_in,
    __nv_bfloat16* __restrict__ state,
    __nv_bfloat16* __restrict__ out_,
    int num_v_heads,
    bool use_qk_l2norm)
{
  static_assert(HD == 128, "HD must be 128 for Qwen3.6 (single instantiation)");
  const int h = blockIdx.x;
  const int b = blockIdx.y;
  const int t = threadIdx.x;
  if (t >= HD) return;

  // Smem layout: qs[HD], ks[HD], scratch[8] (warp-reduce buffer).
  __shared__ float smem[2 * HD + 32];
  float* qs = smem;
  float* ks = smem + HD;
  float* scratch = smem + 2 * HD;

  // Load Q and K to smem (each thread loads its element).
  const size_t qkv_off = ((size_t)b * num_v_heads + h) * HD + t;
  qs[t] = static_cast<float>(q_in[qkv_off]);
  ks[t] = static_cast<float>(k_in[qkv_off]);
  __syncthreads();

  // L2 norm Q and K (in-place in smem).
  if (use_qk_l2norm) {
    float q_sq = qs[t] * qs[t];
    float k_sq = ks[t] * ks[t];
    q_sq = block_reduce_sum<HD>(q_sq, scratch);
    // Required between consecutive block_reduce_sum calls that share
    // the same ``scratch`` smem region — see chunked kernel comment.
    __syncthreads();
    k_sq = block_reduce_sum<HD>(k_sq, scratch);
    const float q_inv = rsqrtf(q_sq + kEps);
    const float k_inv = rsqrtf(k_sq + kEps);
    qs[t] *= q_inv;
    ks[t] *= k_inv;
    __syncthreads();
  }

  // Scale Q by 1 / sqrt(HD).
  qs[t] *= rsqrtf(static_cast<float>(HD));
  __syncthreads();

  // exp(g_t) and beta_t (broadcast scalars).
  const float g_t =
      __expf(static_cast<float>(g_in[b * num_v_heads + h]));
  const float beta_t =
      static_cast<float>(beta_in[b * num_v_heads + h]);

  // Each thread holds column t of state[b, h, :, :] in registers.
  float col[HD];
  const size_t state_h_off =
      (((size_t)b * num_v_heads + h)) * HD * HD;
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    col[i] =
        static_cast<float>(state[state_h_off + (size_t)i * HD + t]) * g_t;
  }

  // kv_mem[t] = sum_i col[i] * ks[i]
  float kv_mem = 0.0f;
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    kv_mem = fmaf(col[i], ks[i], kv_mem);
  }

  // delta[t] = (V[t] - kv_mem) * beta
  const float v_t =
      static_cast<float>(v_in[(size_t)b * num_v_heads * HD + h * HD + t]);
  const float delta = (v_t - kv_mem) * beta_t;

  // state[i, t] += k[i] * delta
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    col[i] = fmaf(ks[i], delta, col[i]);
  }

  // Write back state column.
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    state[state_h_off + (size_t)i * HD + t] = __float2bfloat16(col[i]);
  }

  // out[t] = sum_i col[i] * qs[i]
  float out_t = 0.0f;
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    out_t = fmaf(col[i], qs[i], out_t);
  }
  out_[(size_t)b * num_v_heads * HD + h * HD + t] =
      __float2bfloat16(out_t);
}

}  // namespace

void gated_deltanet_recurrent_qwen36_bf16(
    const void* q,
    const void* k,
    const void* v,
    const void* g,
    const void* beta,
    void*       state,
    void*       out,
    int B, int num_v_heads, int head_k_dim, int head_v_dim,
    bool use_qk_l2norm,
    cudaStream_t stream)
{
  if (head_k_dim != kHD || head_v_dim != kHD) {
    // Could template more dims; for Qwen3.6 only HD=128 is needed.
    return;  // silently no-op; caller checks output is unchanged
  }

  dim3 grid(num_v_heads, B);
  dim3 block(kHD);
  gated_deltanet_recurrent_kernel<kHD><<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q),
      reinterpret_cast<const __nv_bfloat16*>(k),
      reinterpret_cast<const __nv_bfloat16*>(v),
      reinterpret_cast<const __nv_bfloat16*>(g),
      reinterpret_cast<const __nv_bfloat16*>(beta),
      reinterpret_cast<__nv_bfloat16*>(state),
      reinterpret_cast<__nv_bfloat16*>(out),
      num_v_heads, use_qk_l2norm);
}

// In/out-state variant: reads col from state_in, writes updated col
// to state_out (separate buffer). Eliminates the standalone
// .copy_(state_out, state) launch in the K-iter verify loop by
// chaining state_in[k+1] := state_out[k] across iterations. Bit-
// identical to (existing kernel + .copy_) under same inputs because
// the math is unchanged; only the writeback target differs.
namespace {

template <int HD>
__global__ void gated_deltanet_recurrent_inout_kernel(
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ v_in,
    const __nv_bfloat16* __restrict__ g_in,
    const __nv_bfloat16* __restrict__ beta_in,
    const __nv_bfloat16* __restrict__ state_in,
    __nv_bfloat16* __restrict__ state_out,
    __nv_bfloat16* __restrict__ out_,
    int num_v_heads,
    bool use_qk_l2norm)
{
  static_assert(HD == 128, "HD must be 128 for Qwen3.6");
  const int h = blockIdx.x;
  const int b = blockIdx.y;
  const int t = threadIdx.x;
  if (t >= HD) return;

  __shared__ float smem[2 * HD + 32];
  float* qs = smem;
  float* ks = smem + HD;
  float* scratch = smem + 2 * HD;

  const size_t qkv_off = ((size_t)b * num_v_heads + h) * HD + t;
  qs[t] = static_cast<float>(q_in[qkv_off]);
  ks[t] = static_cast<float>(k_in[qkv_off]);
  __syncthreads();

  if (use_qk_l2norm) {
    float q_sq = qs[t] * qs[t];
    float k_sq = ks[t] * ks[t];
    q_sq = block_reduce_sum<HD>(q_sq, scratch);
    // Required between consecutive block_reduce_sum calls that share
    // the same ``scratch`` smem region — see chunked kernel comment.
    __syncthreads();
    k_sq = block_reduce_sum<HD>(k_sq, scratch);
    const float q_inv = rsqrtf(q_sq + kEps);
    const float k_inv = rsqrtf(k_sq + kEps);
    qs[t] *= q_inv;
    ks[t] *= k_inv;
    __syncthreads();
  }

  qs[t] *= rsqrtf(static_cast<float>(HD));
  __syncthreads();

  const float g_t =
      __expf(static_cast<float>(g_in[b * num_v_heads + h]));
  const float beta_t =
      static_cast<float>(beta_in[b * num_v_heads + h]);

  float col[HD];
  const size_t state_h_off =
      (((size_t)b * num_v_heads + h)) * HD * HD;
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    col[i] =
        static_cast<float>(state_in[state_h_off + (size_t)i * HD + t]) * g_t;
  }

  float kv_mem = 0.0f;
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    kv_mem = fmaf(col[i], ks[i], kv_mem);
  }

  const float v_t =
      static_cast<float>(v_in[(size_t)b * num_v_heads * HD + h * HD + t]);
  const float delta = (v_t - kv_mem) * beta_t;

  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    col[i] = fmaf(ks[i], delta, col[i]);
  }

  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    state_out[state_h_off + (size_t)i * HD + t] = __float2bfloat16(col[i]);
  }

  float out_t = 0.0f;
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    out_t = fmaf(col[i], qs[i], out_t);
  }
  out_[(size_t)b * num_v_heads * HD + h * HD + t] =
      __float2bfloat16(out_t);
}

}  // namespace

void gated_deltanet_recurrent_inout_qwen36_bf16(
    const void* q,
    const void* k,
    const void* v,
    const void* g,
    const void* beta,
    const void* state_in,
    void*       state_out,
    void*       out,
    int B, int num_v_heads, int head_k_dim, int head_v_dim,
    bool use_qk_l2norm,
    cudaStream_t stream)
{
  if (head_k_dim != kHD || head_v_dim != kHD) {
    return;
  }
  dim3 grid(num_v_heads, B);
  dim3 block(kHD);
  gated_deltanet_recurrent_inout_kernel<kHD><<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q),
      reinterpret_cast<const __nv_bfloat16*>(k),
      reinterpret_cast<const __nv_bfloat16*>(v),
      reinterpret_cast<const __nv_bfloat16*>(g),
      reinterpret_cast<const __nv_bfloat16*>(beta),
      reinterpret_cast<const __nv_bfloat16*>(state_in),
      reinterpret_cast<__nv_bfloat16*>(state_out),
      reinterpret_cast<__nv_bfloat16*>(out),
      num_v_heads, use_qk_l2norm);
}

// FP32-log-decay variant of the in/out step. Identical math to
// gated_deltanet_recurrent_inout_kernel except that ``g`` is read as
// FP32: the official cached-decode host exposes the log-decay in FP32,
// and rounding it through BF16 (or paying a cast kernel in the swapped
// hot path) is exactly what the structure qualification refused.
namespace {

template <int HD>
__global__ void gated_deltanet_recurrent_inout_gf32_kernel(
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ v_in,
    const float*         __restrict__ g_in,
    const __nv_bfloat16* __restrict__ beta_in,
    const __nv_bfloat16* __restrict__ state_in,
    __nv_bfloat16* __restrict__ state_out,
    __nv_bfloat16* __restrict__ out_,
    int num_v_heads,
    bool use_qk_l2norm)
{
  static_assert(HD == 128, "HD must be 128 for Qwen3.6");
  const int h = blockIdx.x;
  const int b = blockIdx.y;
  const int t = threadIdx.x;
  if (t >= HD) return;

  __shared__ float smem[2 * HD + 32];
  float* qs = smem;
  float* ks = smem + HD;
  float* scratch = smem + 2 * HD;

  const size_t qkv_off = ((size_t)b * num_v_heads + h) * HD + t;
  qs[t] = static_cast<float>(q_in[qkv_off]);
  ks[t] = static_cast<float>(k_in[qkv_off]);
  __syncthreads();

  if (use_qk_l2norm) {
    float q_sq = qs[t] * qs[t];
    float k_sq = ks[t] * ks[t];
    q_sq = block_reduce_sum<HD>(q_sq, scratch);
    // Required between consecutive block_reduce_sum calls that share
    // the same ``scratch`` smem region — see chunked kernel comment.
    __syncthreads();
    k_sq = block_reduce_sum<HD>(k_sq, scratch);
    const float q_inv = rsqrtf(q_sq + kEps);
    const float k_inv = rsqrtf(k_sq + kEps);
    qs[t] *= q_inv;
    ks[t] *= k_inv;
    __syncthreads();
  }

  qs[t] *= rsqrtf(static_cast<float>(HD));
  __syncthreads();

  const float g_t = __expf(g_in[b * num_v_heads + h]);
  const float beta_t =
      static_cast<float>(beta_in[b * num_v_heads + h]);

  float col[HD];
  const size_t state_h_off =
      (((size_t)b * num_v_heads + h)) * HD * HD;
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    col[i] =
        static_cast<float>(state_in[state_h_off + (size_t)i * HD + t]) * g_t;
  }

  float kv_mem = 0.0f;
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    kv_mem = fmaf(col[i], ks[i], kv_mem);
  }

  const float v_t =
      static_cast<float>(v_in[(size_t)b * num_v_heads * HD + h * HD + t]);
  const float delta = (v_t - kv_mem) * beta_t;

  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    col[i] = fmaf(ks[i], delta, col[i]);
  }

  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    state_out[state_h_off + (size_t)i * HD + t] = __float2bfloat16(col[i]);
  }

  float out_t = 0.0f;
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    out_t = fmaf(col[i], qs[i], out_t);
  }
  out_[(size_t)b * num_v_heads * HD + h * HD + t] =
      __float2bfloat16(out_t);
}

}  // namespace

void gated_deltanet_recurrent_inout_qwen36_gf32(
    const void* q,
    const void* k,
    const void* v,
    const void* g,
    const void* beta,
    const void* state_in,
    void*       state_out,
    void*       out,
    int B, int num_v_heads, int head_k_dim, int head_v_dim,
    bool use_qk_l2norm,
    cudaStream_t stream)
{
  if (head_k_dim != kHD || head_v_dim != kHD) {
    return;
  }
  dim3 grid(num_v_heads, B);
  dim3 block(kHD);
  gated_deltanet_recurrent_inout_gf32_kernel<kHD><<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q),
      reinterpret_cast<const __nv_bfloat16*>(k),
      reinterpret_cast<const __nv_bfloat16*>(v),
      reinterpret_cast<const float*>(g),
      reinterpret_cast<const __nv_bfloat16*>(beta),
      reinterpret_cast<const __nv_bfloat16*>(state_in),
      reinterpret_cast<__nv_bfloat16*>(state_out),
      reinterpret_cast<__nv_bfloat16*>(out),
      num_v_heads, use_qk_l2norm);
}

// FP32-log-decay + FP32-state in/out variant: the transformers cached
// decode of this family runs its fallback recurrence entirely in FP32
// and carries the FP32 final state in the cache. Serving it faithfully
// needs g read as float AND the state read/written as float with an
// explicit output — no per-step BF16 rounding of the state, which is
// exactly the host's own numerics.
namespace {

template <int HD>
__global__ void gated_deltanet_recurrent_inout_gf32_sf32_kernel(
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ v_in,
    const float*         __restrict__ g_in,
    const __nv_bfloat16* __restrict__ beta_in,
    const float*         __restrict__ state_in,
    float*               __restrict__ state_out,
    __nv_bfloat16* __restrict__ out_,
    int num_v_heads,
    bool use_qk_l2norm)
{
  static_assert(HD == 128, "HD must be 128 for Qwen3.6");
  const int h = blockIdx.x;
  const int b = blockIdx.y;
  const int t = threadIdx.x;
  if (t >= HD) return;

  __shared__ float smem[2 * HD + 32];
  float* qs = smem;
  float* ks = smem + HD;
  float* scratch = smem + 2 * HD;

  const size_t qkv_off = ((size_t)b * num_v_heads + h) * HD + t;
  qs[t] = static_cast<float>(q_in[qkv_off]);
  ks[t] = static_cast<float>(k_in[qkv_off]);
  __syncthreads();

  if (use_qk_l2norm) {
    float q_sq = qs[t] * qs[t];
    float k_sq = ks[t] * ks[t];
    q_sq = block_reduce_sum<HD>(q_sq, scratch);
    __syncthreads();
    k_sq = block_reduce_sum<HD>(k_sq, scratch);
    const float q_inv = rsqrtf(q_sq + kEps);
    const float k_inv = rsqrtf(k_sq + kEps);
    qs[t] *= q_inv;
    ks[t] *= k_inv;
    __syncthreads();
  }

  qs[t] *= rsqrtf(static_cast<float>(HD));
  __syncthreads();

  const float g_t = __expf(g_in[b * num_v_heads + h]);
  const float beta_t =
      static_cast<float>(beta_in[b * num_v_heads + h]);

  float col[HD];
  const size_t state_h_off =
      (((size_t)b * num_v_heads + h)) * HD * HD;
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    col[i] = state_in[state_h_off + (size_t)i * HD + t] * g_t;
  }

  float kv_mem = 0.0f;
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    kv_mem = fmaf(col[i], ks[i], kv_mem);
  }

  const float v_t =
      static_cast<float>(v_in[(size_t)b * num_v_heads * HD + h * HD + t]);
  const float delta = (v_t - kv_mem) * beta_t;

  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    col[i] = fmaf(ks[i], delta, col[i]);
  }

  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    state_out[state_h_off + (size_t)i * HD + t] = col[i];
  }

  float out_t = 0.0f;
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    out_t = fmaf(col[i], qs[i], out_t);
  }
  out_[(size_t)b * num_v_heads * HD + h * HD + t] =
      __float2bfloat16(out_t);
}

}  // namespace

void gated_deltanet_recurrent_inout_gf32_sf32(
    const void* q,
    const void* k,
    const void* v,
    const void* g,
    const void* beta,
    const void* state_in,
    void*       state_out,
    void*       out,
    int B, int num_v_heads, int head_k_dim, int head_v_dim,
    bool use_qk_l2norm,
    cudaStream_t stream)
{
  if (head_k_dim != kHD || head_v_dim != kHD) {
    return;
  }
  dim3 grid(num_v_heads, B);
  dim3 block(kHD);
  gated_deltanet_recurrent_inout_gf32_sf32_kernel<kHD><<<grid, block, 0,
                                                         stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q),
      reinterpret_cast<const __nv_bfloat16*>(k),
      reinterpret_cast<const __nv_bfloat16*>(v),
      reinterpret_cast<const float*>(g),
      reinterpret_cast<const __nv_bfloat16*>(beta),
      reinterpret_cast<const float*>(state_in),
      reinterpret_cast<float*>(state_out),
      reinterpret_cast<__nv_bfloat16*>(out),
      num_v_heads, use_qk_l2norm);
}

// FP32-state variant. Mathematically identical to the BF16-state path
// (FP32 col[] accumulator), but the persistent state is read AND
// written in FP32 — no __float2bfloat16 round-trip per recurrent
// step. Eliminates the LSB-jitter that accumulates over many
// recurrent iterations and makes K-row prefill diverge from
// per-token at K beyond ~22 on the Thor BF16-state path.
namespace {

template <int HD>
__global__ void gated_deltanet_recurrent_f32state_kernel(
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ v_in,
    const __nv_bfloat16* __restrict__ g_in,
    const __nv_bfloat16* __restrict__ beta_in,
    float*                __restrict__ state,
    __nv_bfloat16* __restrict__ out_,
    int num_v_heads,
    bool use_qk_l2norm)
{
  static_assert(HD == 128, "HD must be 128 for Qwen3.6");
  const int h = blockIdx.x;
  const int b = blockIdx.y;
  const int t = threadIdx.x;
  if (t >= HD) return;

  __shared__ float smem[2 * HD + 32];
  float* qs = smem;
  float* ks = smem + HD;
  float* scratch = smem + 2 * HD;

  const size_t qkv_off = ((size_t)b * num_v_heads + h) * HD + t;
  qs[t] = static_cast<float>(q_in[qkv_off]);
  ks[t] = static_cast<float>(k_in[qkv_off]);
  __syncthreads();

  if (use_qk_l2norm) {
    float q_sq = qs[t] * qs[t];
    float k_sq = ks[t] * ks[t];
    q_sq = block_reduce_sum<HD>(q_sq, scratch);
    // Required between consecutive block_reduce_sum calls that share
    // the same ``scratch`` smem region — see chunked kernel comment.
    __syncthreads();
    k_sq = block_reduce_sum<HD>(k_sq, scratch);
    const float q_inv = rsqrtf(q_sq + kEps);
    const float k_inv = rsqrtf(k_sq + kEps);
    qs[t] *= q_inv;
    ks[t] *= k_inv;
    __syncthreads();
  }

  qs[t] *= rsqrtf(static_cast<float>(HD));
  __syncthreads();

  const float g_t =
      __expf(static_cast<float>(g_in[b * num_v_heads + h]));
  const float beta_t =
      static_cast<float>(beta_in[b * num_v_heads + h]);

  float col[HD];
  const size_t state_h_off =
      (((size_t)b * num_v_heads + h)) * HD * HD;
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    col[i] = state[state_h_off + (size_t)i * HD + t] * g_t;
  }

  float kv_mem = 0.0f;
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    kv_mem = fmaf(col[i], ks[i], kv_mem);
  }

  const float v_t =
      static_cast<float>(v_in[(size_t)b * num_v_heads * HD + h * HD + t]);
  const float delta = (v_t - kv_mem) * beta_t;

  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    col[i] = fmaf(ks[i], delta, col[i]);
  }

  // FP32 state write — no rounding.
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    state[state_h_off + (size_t)i * HD + t] = col[i];
  }

  float out_t = 0.0f;
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    out_t = fmaf(col[i], qs[i], out_t);
  }
  out_[(size_t)b * num_v_heads * HD + h * HD + t] =
      __float2bfloat16(out_t);
}

}  // namespace

void gated_deltanet_recurrent_qwen36_f32state_bf16io(
    const void* q,
    const void* k,
    const void* v,
    const void* g,
    const void* beta,
    void*       state_f32,
    void*       out,
    int B, int num_v_heads, int head_k_dim, int head_v_dim,
    bool use_qk_l2norm,
    cudaStream_t stream)
{
  if (head_k_dim != kHD || head_v_dim != kHD) {
    return;
  }
  dim3 grid(num_v_heads, B);
  dim3 block(kHD);
  gated_deltanet_recurrent_f32state_kernel<kHD><<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q),
      reinterpret_cast<const __nv_bfloat16*>(k),
      reinterpret_cast<const __nv_bfloat16*>(v),
      reinterpret_cast<const __nv_bfloat16*>(g),
      reinterpret_cast<const __nv_bfloat16*>(beta),
      reinterpret_cast<float*>(state_f32),
      reinterpret_cast<__nv_bfloat16*>(out),
      num_v_heads, use_qk_l2norm);
}

namespace {

template <int HD>
__global__ void gated_deltanet_chunk_kernel(
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ v_in,
    const __nv_bfloat16* __restrict__ g_in,
    const __nv_bfloat16* __restrict__ beta_in,
    __nv_bfloat16* __restrict__ state,
    __nv_bfloat16* __restrict__ out_,
    int S,
    int num_v_heads,
    bool use_qk_l2norm)
{
  static_assert(HD == 128, "HD must be 128 for Qwen3.6");
  const int h = blockIdx.x;
  const int b = blockIdx.y;
  const int t = threadIdx.x;
  if (t >= HD) return;

  __shared__ float smem[2 * HD + 32];
  float* qs = smem;
  float* ks = smem + HD;
  float* scratch = smem + 2 * HD;

  const size_t state_h_off =
      (((size_t)b * num_v_heads + h)) * HD * HD;
  float col[HD];
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    col[i] = static_cast<float>(
        state[state_h_off + (size_t)i * HD + t]);
  }

  for (int s = 0; s < S; ++s) {
    const size_t qkv_off = ((size_t)s * num_v_heads + h) * HD + t;
    qs[t] = static_cast<float>(q_in[qkv_off]);
    ks[t] = static_cast<float>(k_in[qkv_off]);
    __syncthreads();

    if (use_qk_l2norm) {
      float q_sq = qs[t] * qs[t];
      float k_sq = ks[t] * ks[t];
      q_sq = block_reduce_sum<HD>(q_sq, scratch);
      // Required between consecutive block_reduce_sum calls that share
      // the same ``scratch`` smem region. Without this barrier, warp 0
      // can begin writing scratch[warp] inside the second call while a
      // slower warp is still reading scratch[0] (the first call's
      // result) from ``return smem[0]`` — a write-after-read race
      // that produces ~1% non-deterministic output at S>=512.
      __syncthreads();
      k_sq = block_reduce_sum<HD>(k_sq, scratch);
      const float q_inv = rsqrtf(q_sq + kEps);
      const float k_inv = rsqrtf(k_sq + kEps);
      qs[t] *= q_inv;
      ks[t] *= k_inv;
      __syncthreads();
    }

    qs[t] *= rsqrtf(static_cast<float>(HD));
    __syncthreads();

    const float g_t =
        __expf(static_cast<float>(g_in[s * num_v_heads + h]));
    const float beta_t =
        static_cast<float>(beta_in[s * num_v_heads + h]);

    #pragma unroll 16
    for (int i = 0; i < HD; ++i) {
      col[i] *= g_t;
    }

    float kv_mem = 0.0f;
    #pragma unroll 16
    for (int i = 0; i < HD; ++i) {
      kv_mem = fmaf(col[i], ks[i], kv_mem);
    }

    const float v_t = static_cast<float>(v_in[qkv_off]);
    const float delta = (v_t - kv_mem) * beta_t;

    #pragma unroll 16
    for (int i = 0; i < HD; ++i) {
      col[i] = fmaf(ks[i], delta, col[i]);
    }

    float out_t = 0.0f;
    #pragma unroll 16
    for (int i = 0; i < HD; ++i) {
      out_t = fmaf(col[i], qs[i], out_t);
    }
    out_[qkv_off] = __float2bfloat16(out_t);

    // Serial decode stores bf16 state after each token, so the next
    // token reads bf16-quantized state. Mirror that quantization point
    // without paying global memory traffic for intermediate states.
    if (s + 1 < S) {
      #pragma unroll 16
      for (int i = 0; i < HD; ++i) {
        col[i] = static_cast<float>(__float2bfloat16(col[i]));
      }
    }
    __syncthreads();
  }

  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    state[state_h_off + (size_t)i * HD + t] = __float2bfloat16(col[i]);
  }
}

template <int HD>
__global__ void gated_deltanet_chunk_smem_kernel(
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ v_in,
    const __nv_bfloat16* __restrict__ g_in,
    const __nv_bfloat16* __restrict__ beta_in,
    __nv_bfloat16* __restrict__ state,
    __nv_bfloat16* __restrict__ out_,
    int S,
    int num_v_heads,
    bool use_qk_l2norm)
{
  static_assert(HD == 128, "HD must be 128 for Qwen3.6");
  const int h = blockIdx.x;
  const int b = blockIdx.y;
  const int t = threadIdx.x;
  if (t >= HD) return;

  extern __shared__ float smem[];
  float* state_s = smem;
  float* qs = state_s + HD * HD;
  float* ks = qs + HD;
  float* scratch = ks + HD;

  const size_t state_h_off =
      (((size_t)b * num_v_heads + h)) * HD * HD;
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    state_s[i * HD + t] = static_cast<float>(
        state[state_h_off + (size_t)i * HD + t]);
  }
  __syncthreads();

  for (int s = 0; s < S; ++s) {
    const size_t qkv_off = ((size_t)s * num_v_heads + h) * HD + t;
    qs[t] = static_cast<float>(q_in[qkv_off]);
    ks[t] = static_cast<float>(k_in[qkv_off]);
    __syncthreads();

    if (use_qk_l2norm) {
      float q_sq = qs[t] * qs[t];
      float k_sq = ks[t] * ks[t];
      q_sq = block_reduce_sum<HD>(q_sq, scratch);
      // Required between consecutive block_reduce_sum calls that share
      // the same ``scratch`` smem region. Without this barrier, warp 0
      // can begin writing scratch[warp] inside the second call while a
      // slower warp is still reading scratch[0] (the first call's
      // result) from ``return smem[0]`` — a write-after-read race
      // that produces ~1% non-deterministic output at S>=512.
      __syncthreads();
      k_sq = block_reduce_sum<HD>(k_sq, scratch);
      const float q_inv = rsqrtf(q_sq + kEps);
      const float k_inv = rsqrtf(k_sq + kEps);
      qs[t] *= q_inv;
      ks[t] *= k_inv;
      __syncthreads();
    }

    qs[t] *= rsqrtf(static_cast<float>(HD));
    __syncthreads();

    const float g_t =
        __expf(static_cast<float>(g_in[s * num_v_heads + h]));
    const float beta_t =
        static_cast<float>(beta_in[s * num_v_heads + h]);

    #pragma unroll 16
    for (int i = 0; i < HD; ++i) {
      state_s[i * HD + t] *= g_t;
    }

    float kv_mem = 0.0f;
    #pragma unroll 16
    for (int i = 0; i < HD; ++i) {
      kv_mem = fmaf(state_s[i * HD + t], ks[i], kv_mem);
    }

    const float v_t = static_cast<float>(v_in[qkv_off]);
    const float delta = (v_t - kv_mem) * beta_t;

    #pragma unroll 16
    for (int i = 0; i < HD; ++i) {
      state_s[i * HD + t] =
          fmaf(ks[i], delta, state_s[i * HD + t]);
    }

    float out_t = 0.0f;
    #pragma unroll 16
    for (int i = 0; i < HD; ++i) {
      out_t = fmaf(state_s[i * HD + t], qs[i], out_t);
    }
    out_[qkv_off] = __float2bfloat16(out_t);

    if (s + 1 < S) {
      #pragma unroll 16
      for (int i = 0; i < HD; ++i) {
        state_s[i * HD + t] =
            static_cast<float>(__float2bfloat16(state_s[i * HD + t]));
      }
    }
    __syncthreads();
  }

  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    state[state_h_off + (size_t)i * HD + t] =
        __float2bfloat16(state_s[i * HD + t]);
  }
}

__global__ void lin_split_qkv_broadcast_h_kernel(
    const __nv_bfloat16* __restrict__ conv_out,
    __nv_bfloat16* __restrict__ q48,
    __nv_bfloat16* __restrict__ k48,
    __nv_bfloat16* __restrict__ v48,
    int S,
    int num_v_heads,
    int num_k_heads,
    int head_dim)
{
  const int idx = blockIdx.x * kSplitThreads + threadIdx.x;
  const int total = S * num_v_heads * head_dim;
  if (idx >= total) return;

  const int t = idx % head_dim;
  const int h = (idx / head_dim) % num_v_heads;
  const int s = idx / (num_v_heads * head_dim);
  const int broadcast = num_v_heads / num_k_heads;
  const int src_h = h / broadcast;
  const int qk_width = num_k_heads * head_dim;
  const int row_width = (2 * num_k_heads + num_v_heads) * head_dim;
  const size_t row = static_cast<size_t>(s) * row_width;
  q48[idx] = conv_out[row + src_h * head_dim + t];
  k48[idx] = conv_out[row + qk_width + src_h * head_dim + t];
  v48[idx] = conv_out[row + 2 * qk_width + h * head_dim + t];
}

__global__ void qwen36_lin_split_qkv_gqa_kernel(
    const __nv_bfloat16* __restrict__ conv_out,
    __nv_bfloat16* __restrict__ q16,
    __nv_bfloat16* __restrict__ k16,
    __nv_bfloat16* __restrict__ v48,
    int S)
{
  const int idx = blockIdx.x * kSplitThreads + threadIdx.x;
  const int total = S * 10240;
  if (idx >= total) return;

  const int col = idx % 10240;
  const int row = idx / 10240;
  const __nv_bfloat16 x = conv_out[idx];
  if (col < 2048) {
    q16[static_cast<size_t>(row) * 2048 + col] = x;
  } else if (col < 4096) {
    k16[static_cast<size_t>(row) * 2048 + (col - 2048)] = x;
  } else {
    v48[static_cast<size_t>(row) * 6144 + (col - 4096)] = x;
  }
}

__global__ void qwen36_split_q_gate_kernel(
    const __nv_bfloat16* __restrict__ q_proj,
    __nv_bfloat16* __restrict__ q_pre,
    __nv_bfloat16* __restrict__ gate,
    int S)
{
  const int idx = blockIdx.x * kSplitThreads + threadIdx.x;
  const int total = S * 24 * 256;
  if (idx >= total) return;

  const int t = idx % 256;
  const int h = (idx / 256) % 24;
  const int s = idx / (24 * 256);
  const size_t src = (static_cast<size_t>(s) * 24 + h) * 512 + t;
  q_pre[idx] = q_proj[src];
  gate[idx] = q_proj[src + 256];
}

__global__ void qwen36_gdn_gating_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    const float* __restrict__ neg_exp_A_log,
    const float* __restrict__ dt_bias,
    __nv_bfloat16* __restrict__ g_out,
    __nv_bfloat16* __restrict__ beta_out,
    int S,
    int num_heads)
{
  const int idx = blockIdx.x * kSplitThreads + threadIdx.x;
  const int total = S * num_heads;
  if (idx >= total) return;
  const int h = idx % num_heads;

  const float av = static_cast<float>(a[idx]) + dt_bias[h];
  const float sp = log1pf(__expf(av));
  const float gv = neg_exp_A_log[h] * sp;
  const float bv = static_cast<float>(b[idx]);
  const float beta = 1.0f / (1.0f + __expf(-bv));
  g_out[idx] = __float2bfloat16(gv);
  beta_out[idx] = __float2bfloat16(beta);
}

__global__ void qwen36_gdn_gating_strided_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    const float* __restrict__ neg_exp_A_log,
    const float* __restrict__ dt_bias,
    __nv_bfloat16* __restrict__ g_out,
    __nv_bfloat16* __restrict__ beta_out,
    int S,
    int num_heads,
    int a_stride,
    int b_stride)
{
  const int idx = blockIdx.x * kSplitThreads + threadIdx.x;
  const int total = S * num_heads;
  if (idx >= total) return;
  const int row = idx / num_heads;
  const int h = idx - row * num_heads;

  const float av = static_cast<float>(a[row * a_stride + h]) + dt_bias[h];
  const float sp = log1pf(__expf(av));
  const float gv = neg_exp_A_log[h] * sp;
  const float bv = static_cast<float>(b[row * b_stride + h]);
  const float beta = 1.0f / (1.0f + __expf(-bv));
  g_out[idx] = __float2bfloat16(gv);
  beta_out[idx] = __float2bfloat16(beta);
}

template <int HD>
__global__ void qwen36_gdn_chunk_from_conv_smem_kernel(
    const __nv_bfloat16* __restrict__ conv_out,
    const __nv_bfloat16* __restrict__ a_in,
    const __nv_bfloat16* __restrict__ b_in,
    const float* __restrict__ neg_exp_A_log,
    const float* __restrict__ dt_bias,
    __nv_bfloat16* __restrict__ state,
    __nv_bfloat16* __restrict__ out_,
    int S,
    int num_v_heads,
    int num_k_heads,
    int a_stride,
    int b_stride,
    bool use_qk_l2norm)
{
  static_assert(HD == 128, "HD must be 128 for Qwen3.6");
  const int h = blockIdx.x;
  const int b = blockIdx.y;
  const int t = threadIdx.x;
  if (t >= HD) return;

  extern __shared__ float smem[];
  float* state_s = smem;
  float* qs = state_s + HD * HD;
  float* ks = qs + HD;
  float* scratch = ks + HD;
  float* gate_values = scratch + 32;

  const size_t state_h_off =
      (((size_t)b * num_v_heads + h)) * HD * HD;
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    state_s[i * HD + t] = static_cast<float>(
        state[state_h_off + (size_t)i * HD + t]);
  }
  __syncthreads();

  const int broadcast = num_v_heads / num_k_heads;
  const int src_h = h / broadcast;
  const int qk_width = num_k_heads * HD;
  const int row_width = (2 * num_k_heads + num_v_heads) * HD;
  for (int s = 0; s < S; ++s) {
    const size_t row = static_cast<size_t>(s) * row_width;
    const size_t out_off = ((size_t)s * num_v_heads + h) * HD + t;
    qs[t] = static_cast<float>(conv_out[row + src_h * HD + t]);
    ks[t] = static_cast<float>(conv_out[row + qk_width + src_h * HD + t]);
    __syncthreads();

    if (use_qk_l2norm) {
      float q_sq = qs[t] * qs[t];
      float k_sq = ks[t] * ks[t];
      q_sq = block_reduce_sum<HD>(q_sq, scratch);
      // Required between consecutive block_reduce_sum calls that share
      // the same ``scratch`` smem region. Without this barrier, warp 0
      // can begin writing scratch[warp] inside the second call while a
      // slower warp is still reading scratch[0] (the first call's
      // result) from ``return smem[0]`` — a write-after-read race
      // that produces ~1% non-deterministic output at S>=512.
      __syncthreads();
      k_sq = block_reduce_sum<HD>(k_sq, scratch);
      const float q_inv = rsqrtf(q_sq + kEps);
      const float k_inv = rsqrtf(k_sq + kEps);
      qs[t] *= q_inv;
      ks[t] *= k_inv;
      __syncthreads();
    }

    qs[t] *= rsqrtf(static_cast<float>(HD));
    __syncthreads();

    if (t == 0) {
      const float av =
          static_cast<float>(a_in[s * a_stride + h]) + dt_bias[h];
      const float sp = log1pf(__expf(av));
      const float g_log = static_cast<float>(
          __float2bfloat16(neg_exp_A_log[h] * sp));
      gate_values[0] = __expf(g_log);
      const float bv = static_cast<float>(b_in[s * b_stride + h]);
      gate_values[1] = static_cast<float>(
          __float2bfloat16(1.0f / (1.0f + __expf(-bv))));
    }
    __syncthreads();
    const float g_t = gate_values[0];
    const float beta_t = gate_values[1];

    #pragma unroll 16
    for (int i = 0; i < HD; ++i) {
      state_s[i * HD + t] *= g_t;
    }

    float kv_mem = 0.0f;
    #pragma unroll 16
    for (int i = 0; i < HD; ++i) {
      kv_mem = fmaf(state_s[i * HD + t], ks[i], kv_mem);
    }

    const float v_t =
        static_cast<float>(conv_out[row + 2 * qk_width + h * HD + t]);
    const float delta = (v_t - kv_mem) * beta_t;

    #pragma unroll 16
    for (int i = 0; i < HD; ++i) {
      state_s[i * HD + t] =
          fmaf(ks[i], delta, state_s[i * HD + t]);
    }

    float out_t = 0.0f;
    #pragma unroll 16
    for (int i = 0; i < HD; ++i) {
      out_t = fmaf(state_s[i * HD + t], qs[i], out_t);
    }
    out_[out_off] = __float2bfloat16(out_t);

    if (s + 1 < S) {
      #pragma unroll 16
      for (int i = 0; i < HD; ++i) {
        state_s[i * HD + t] =
            static_cast<float>(__float2bfloat16(state_s[i * HD + t]));
      }
    }
    __syncthreads();
  }

  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    state[state_h_off + (size_t)i * HD + t] =
        __float2bfloat16(state_s[i * HD + t]);
  }
}

__global__ void qwen36_gdn_wy_norm_qk_kernel(
    const __nv_bfloat16* __restrict__ q16,
    const __nv_bfloat16* __restrict__ k16,
    __nv_bfloat16* __restrict__ q16_l2,
    __nv_bfloat16* __restrict__ k16_l2,
    __nv_bfloat16* __restrict__ q_pack_hv,
    __nv_bfloat16* __restrict__ k_pack_hk,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_group_size)
{
  const int t = threadIdx.x;
  const int h = blockIdx.x;
  const int s = blockIdx.y;
  if (t >= kHD || h >= num_k_heads || s >= S) return;

  __shared__ float scratch[32];
  const size_t off = (static_cast<size_t>(s) * num_k_heads + h) * kHD + t;
  const float qv = static_cast<float>(q16[off]);
  const float kv = static_cast<float>(k16[off]);
  float q_sq = qv * qv;
  float k_sq = kv * kv;
  q_sq = block_reduce_sum<kHD>(q_sq, scratch);
  __syncthreads();
  k_sq = block_reduce_sum<kHD>(k_sq, scratch);
  __syncthreads();
  const float q_inv = rsqrtf(q_sq + kEps);
  const float k_inv = rsqrtf(k_sq + kEps);
  const __nv_bfloat16 q_norm = __float2bfloat16(qv * q_inv);
  const __nv_bfloat16 k_norm = __float2bfloat16(kv * k_inv);
  q16_l2[off] = q_norm;
  k16_l2[off] = k_norm;
  if (k_pack_hk != nullptr) {
    const int chunk = s / kWyChunk;
    const int tt = s - chunk * kWyChunk;
    k_pack_hk[
        ((static_cast<size_t>(chunk) * num_k_heads + h) * kWyChunk + tt)
        * kHD + t] = k_norm;
  }
  if (q_pack_hv != nullptr) {
    const int chunk = s / kWyChunk;
    const int tt = s - chunk * kWyChunk;
    #pragma unroll
    for (int r = 0; r < head_group_size; ++r) {
      const int vh = h * head_group_size + r;
      q_pack_hv[
          ((static_cast<size_t>(chunk) * num_v_heads + vh) * kWyChunk + tt)
          * kHD + t] = q_norm;
    }
  }
}

__global__ void qwen36_gdn_wy_cumsum_g_kernel(
    const __nv_bfloat16* __restrict__ g,
    __nv_bfloat16* __restrict__ g_cumsum,
    int S,
    int num_v_heads)
{
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= num_v_heads) return;
  float acc = 0.0f;
  for (int s = 0; s < S; ++s) {
    if ((s % kWyChunk) == 0) {
      acc = 0.0f;
    }
    const size_t off = static_cast<size_t>(s) * num_v_heads + h;
    acc += static_cast<float>(g[off]);
    g_cumsum[off] = __float2bfloat16(acc);
  }
}

__global__ void qwen36_gdn_wy_kkt_b64_kernel(
    const __nv_bfloat16* __restrict__ k16_l2,
    const __nv_bfloat16* __restrict__ beta,
    const __nv_bfloat16* __restrict__ g_cumsum,
    float* __restrict__ A,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_group_size)
{
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair >= kWyChunk * kWyChunk) return;
  const int i = pair / kWyChunk;
  const int j = pair - i * kWyChunk;
  const int vh = blockIdx.y;
  const int chunk = blockIdx.z;
  const int si = chunk * kWyChunk + i;
  const int sj = chunk * kWyChunk + j;
  const size_t a_off =
      (((static_cast<size_t>(chunk) * num_v_heads + vh) * kWyChunk + i)
       * kWyChunk + j);
  if (i <= j || si >= S || sj >= S) {
    A[a_off] = 0.0f;
    return;
  }

  const int kh = vh / head_group_size;
  const size_t ki_base =
      (static_cast<size_t>(si) * num_k_heads + kh) * kHD;
  const size_t kj_base =
      (static_cast<size_t>(sj) * num_k_heads + kh) * kHD;
  float dot = 0.0f;
  #pragma unroll 16
  for (int d = 0; d < kHD; ++d) {
    dot = fmaf(
        static_cast<float>(k16_l2[ki_base + d]),
        static_cast<float>(k16_l2[kj_base + d]),
        dot);
  }
  const float beta_i =
      static_cast<float>(beta[static_cast<size_t>(si) * num_v_heads + vh]);
  const float gi =
      static_cast<float>(g_cumsum[static_cast<size_t>(si) * num_v_heads + vh]);
  const float gj =
      static_cast<float>(g_cumsum[static_cast<size_t>(sj) * num_v_heads + vh]);
  A[a_off] = beta_i * dot * __expf(gi - gj);
}

__global__ void qwen36_gdn_wy_solve_tril_b64_kernel(
    const float* __restrict__ A,
    float* __restrict__ Ai,
    int S,
    int num_v_heads)
{
  const int vh = blockIdx.x;
  const int chunk = blockIdx.y;
  const int base_s = chunk * kWyChunk;
  const size_t base =
      (static_cast<size_t>(chunk) * num_v_heads + vh) * kWyChunk * kWyChunk;
  const int lane = threadIdx.x;
  const int T = min(kWyChunk, S - base_s);
  __shared__ float a_shared[kWyChunk * kWyChunk];
  __shared__ float inv_shared[kWyChunk * kWyChunk];

  for (int idx = lane; idx < kWyChunk * kWyChunk; idx += blockDim.x) {
    a_shared[idx] = A[base + idx];
    const int r = idx / kWyChunk;
    const int c = idx - r * kWyChunk;
    inv_shared[idx] = (r == c && r < T) ? 1.0f : 0.0f;
  }
  __syncthreads();

  // Each column in the current row is independent once prior rows are done.
  // The inner m-loop remains ordered, preserving the production reduction
  // contract while exposing 64-way parallelism across columns.
  for (int r = 1; r < T; ++r) {
    if (lane < r) {
      float val = -a_shared[r * kWyChunk + lane];
      for (int m = lane + 1; m < r; ++m) {
        val -= a_shared[r * kWyChunk + m] *
               inv_shared[m * kWyChunk + lane];
      }
      inv_shared[r * kWyChunk + lane] = val;
    }
    __syncthreads();
  }

  for (int idx = lane; idx < kWyChunk * kWyChunk; idx += blockDim.x) {
    Ai[base + idx] = inv_shared[idx];
  }
}

__global__ void qwen36_gdn_wy_cast_ai_f32_to_bf16_kernel(
    const float* __restrict__ Ai,
    __nv_bfloat16* __restrict__ Ai_pack,
    int total)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) {
    Ai_pack[idx] = __float2bfloat16(Ai[idx]);
  }
}

__global__ void qwen36_gdn_wy_recompute_wu_b64_kernel(
    const __nv_bfloat16* __restrict__ k16_l2,
    const __nv_bfloat16* __restrict__ v48,
    const __nv_bfloat16* __restrict__ beta,
    const __nv_bfloat16* __restrict__ g_cumsum,
    const float* __restrict__ Ai,
    __nv_bfloat16* __restrict__ w48,
    __nv_bfloat16* __restrict__ u48,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_group_size)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = S * num_v_heads * kHD;
  if (idx >= total) return;

  const int d = idx % kHD;
  const int vh = (idx / kHD) % num_v_heads;
  const int s = idx / (num_v_heads * kHD);
  const int chunk = s / kWyChunk;
  const int i = s - chunk * kWyChunk;
  const int kh = vh / head_group_size;
  const int chunk_start = chunk * kWyChunk;
  const int T = min(kWyChunk, S - chunk_start);
  const size_t ai_base =
      (static_cast<size_t>(chunk) * num_v_heads + vh) * kWyChunk * kWyChunk
      + static_cast<size_t>(i) * kWyChunk;

  float u_acc = 0.0f;
  float w_acc = 0.0f;
  for (int j = 0; j < T; ++j) {
    const int sj = chunk_start + j;
    const float aij = Ai[ai_base + j];
    const float beta_j =
        static_cast<float>(beta[static_cast<size_t>(sj) * num_v_heads + vh]);
    const float vj =
        static_cast<float>(v48[(static_cast<size_t>(sj) * num_v_heads + vh)
                               * kHD + d]);
    const float kj =
        static_cast<float>(k16_l2[
            (static_cast<size_t>(sj) * num_k_heads + kh) * kHD + d]);
    const float gj =
        static_cast<float>(g_cumsum[static_cast<size_t>(sj) * num_v_heads + vh]);
    u_acc = fmaf(aij, vj * beta_j, u_acc);
    w_acc = fmaf(aij, kj * beta_j * __expf(gj), w_acc);
  }
  u48[idx] = __float2bfloat16(u_acc);
  w48[idx] = __float2bfloat16(w_acc);
}

__global__ void qwen36_gdn_wy_chunk_h_b64_kernel(
    const __nv_bfloat16* __restrict__ k16_l2,
    const __nv_bfloat16* __restrict__ u48,
    const __nv_bfloat16* __restrict__ w48,
    const __nv_bfloat16* __restrict__ g_cumsum,
    __nv_bfloat16* __restrict__ state,
    __nv_bfloat16* __restrict__ h0,
    __nv_bfloat16* __restrict__ v_new,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_group_size)
{
  const int vh = blockIdx.x;
  const int d = threadIdx.x;
  if (vh >= num_v_heads || d >= kHD) return;

  extern __shared__ float smem[];
  float* state_s = smem;
  const int kh = vh / head_group_size;
  const int chunks = (S + kWyChunk - 1) / kWyChunk;
  const size_t state_base = static_cast<size_t>(vh) * kHD * kHD;

  #pragma unroll 16
  for (int r = 0; r < kHD; ++r) {
    state_s[r * kHD + d] =
        static_cast<float>(state[state_base + static_cast<size_t>(r) * kHD + d]);
  }
  __syncthreads();

  float vbuf[kWyChunk];
  for (int ci = 0; ci < chunks; ++ci) {
    const int start = ci * kWyChunk;
    const int T = min(kWyChunk, S - start);
    const size_t h_base =
        (static_cast<size_t>(ci) * num_v_heads + vh) * kHD * kHD;

    #pragma unroll 16
    for (int r = 0; r < kHD; ++r) {
      h0[h_base + static_cast<size_t>(r) * kHD + d] =
          __float2bfloat16(state_s[r * kHD + d]);
    }
    __syncthreads();

    for (int t = 0; t < kWyChunk; ++t) {
      float val = 0.0f;
      if (t < T) {
        const int s = start + t;
        const size_t wh_base =
            (static_cast<size_t>(s) * num_v_heads + vh) * kHD;
        #pragma unroll 16
        for (int r = 0; r < kHD; ++r) {
          val = fmaf(static_cast<float>(w48[wh_base + r]),
                     state_s[r * kHD + d], val);
        }
        val = static_cast<float>(u48[wh_base + d]) - val;
        v_new[wh_base + d] = __float2bfloat16(val);
      }
      vbuf[t] = val;
    }
    __syncthreads();

    if (T > 0) {
      const float g_last =
          static_cast<float>(g_cumsum[
              static_cast<size_t>(start + T - 1) * num_v_heads + vh]);
      const float eg_last = __expf(g_last);
      #pragma unroll 16
      for (int r = 0; r < kHD; ++r) {
        state_s[r * kHD + d] *= eg_last;
      }
      __syncthreads();

      #pragma unroll 16
      for (int r = 0; r < kHD; ++r) {
        float acc = state_s[r * kHD + d];
        for (int t = 0; t < T; ++t) {
          const int s = start + t;
          const float gt =
              static_cast<float>(g_cumsum[static_cast<size_t>(s) * num_v_heads + vh]);
          const float decay = __expf(g_last - gt);
          const float kval = static_cast<float>(
              k16_l2[(static_cast<size_t>(s) * num_k_heads + kh) * kHD + r]);
          acc = fmaf(kval, vbuf[t] * decay, acc);
        }
        state_s[r * kHD + d] = acc;
      }
      __syncthreads();
    }
  }

  #pragma unroll 16
  for (int r = 0; r < kHD; ++r) {
    state[state_base + static_cast<size_t>(r) * kHD + d] =
        __float2bfloat16(state_s[r * kHD + d]);
  }
}

__global__ void qwen36_gdn_wy_output_o_b64_kernel(
    const __nv_bfloat16* __restrict__ q16_l2,
    const __nv_bfloat16* __restrict__ k16_l2,
    const __nv_bfloat16* __restrict__ v_new,
    const __nv_bfloat16* __restrict__ h0,
    const __nv_bfloat16* __restrict__ g_cumsum,
    __nv_bfloat16* __restrict__ out,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_group_size)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = S * num_v_heads * kHD;
  if (idx >= total) return;

  const int d = idx % kHD;
  const int vh = (idx / kHD) % num_v_heads;
  const int s = idx / (num_v_heads * kHD);
  const int kh = vh / head_group_size;
  const int chunk = s / kWyChunk;
  const int i = s - chunk * kWyChunk;
  const int start = chunk * kWyChunk;
  const size_t q_base = (static_cast<size_t>(s) * num_k_heads + kh) * kHD;
  const size_t h_base =
      (static_cast<size_t>(chunk) * num_v_heads + vh) * kHD * kHD;
  const float gi =
      static_cast<float>(g_cumsum[static_cast<size_t>(s) * num_v_heads + vh]);

  float qh = 0.0f;
  #pragma unroll 16
  for (int r = 0; r < kHD; ++r) {
    qh = fmaf(static_cast<float>(q16_l2[q_base + r]),
              static_cast<float>(h0[h_base + static_cast<size_t>(r) * kHD + d]),
              qh);
  }
  qh *= __expf(gi);

  float local = 0.0f;
  for (int tj = 0; tj <= i; ++tj) {
    const int sj = start + tj;
    if (sj >= S) break;
    const size_t kj_base =
        (static_cast<size_t>(sj) * num_k_heads + kh) * kHD;
    float qk = 0.0f;
    #pragma unroll 16
    for (int r = 0; r < kHD; ++r) {
      qk = fmaf(static_cast<float>(q16_l2[q_base + r]),
                static_cast<float>(k16_l2[kj_base + r]), qk);
    }
    const float gj =
        static_cast<float>(
            g_cumsum[static_cast<size_t>(sj) * num_v_heads + vh]);
    const float vv =
        static_cast<float>(v_new[(static_cast<size_t>(sj) * num_v_heads + vh)
                                 * kHD + d]);
    local = fmaf(qk * __expf(gi - gj), vv, local);
  }

  constexpr float kScale = 0.08838834764831845f;  // 1 / sqrt(128)
  out[idx] = __float2bfloat16((qh + local) * kScale);
}

}  // namespace

void gated_deltanet_chunk_qwen36_bf16(
    const void* q,
    const void* k,
    const void* v,
    const void* g,
    const void* beta,
    void*       state,
    void*       out,
    int S, int num_v_heads, int head_k_dim, int head_v_dim,
    bool use_qk_l2norm,
    cudaStream_t stream)
{
  if (head_k_dim != kHD || head_v_dim != kHD || S <= 0) {
    return;
  }
  dim3 grid(num_v_heads, 1);
  dim3 block(kHD);
  gated_deltanet_chunk_kernel<kHD><<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q),
      reinterpret_cast<const __nv_bfloat16*>(k),
      reinterpret_cast<const __nv_bfloat16*>(v),
      reinterpret_cast<const __nv_bfloat16*>(g),
      reinterpret_cast<const __nv_bfloat16*>(beta),
      reinterpret_cast<__nv_bfloat16*>(state),
      reinterpret_cast<__nv_bfloat16*>(out),
      S, num_v_heads, use_qk_l2norm);
}

void qwen36_lin_split_qkv_broadcast_bf16(
    const void* conv_out,
    void*       q48,
    void*       k48,
    void*       v48,
    int S,
    cudaStream_t stream)
{
  lin_split_qkv_broadcast_h_bf16(
      conv_out, q48, k48, v48, S, 48, 16, kHD, stream);
}

void lin_split_qkv_broadcast_h_bf16(
    const void* conv_out,
    void*       q,
    void*       k,
    void*       v,
    int S,
    int num_v_heads,
    int num_k_heads,
    int head_dim,
    cudaStream_t stream)
{
  if (S <= 0) return;
  const int total = S * num_v_heads * head_dim;
  dim3 grid((total + kSplitThreads - 1) / kSplitThreads);
  dim3 block(kSplitThreads);
  lin_split_qkv_broadcast_h_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(conv_out),
      reinterpret_cast<__nv_bfloat16*>(q),
      reinterpret_cast<__nv_bfloat16*>(k),
      reinterpret_cast<__nv_bfloat16*>(v),
      S, num_v_heads, num_k_heads, head_dim);
}

void qwen36_lin_split_qkv_gqa_bf16(
    const void* conv_out,
    void*       q16,
    void*       k16,
    void*       v48,
    int S,
    cudaStream_t stream)
{
  if (S <= 0) return;
  const int total = S * 10240;
  dim3 grid((total + kSplitThreads - 1) / kSplitThreads);
  dim3 block(kSplitThreads);
  qwen36_lin_split_qkv_gqa_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(conv_out),
      reinterpret_cast<__nv_bfloat16*>(q16),
      reinterpret_cast<__nv_bfloat16*>(k16),
      reinterpret_cast<__nv_bfloat16*>(v48),
      S);
}

void qwen36_split_q_gate_bf16(
    const void* q_proj,
    void*       q_pre,
    void*       gate,
    int S,
    cudaStream_t stream)
{
  if (S <= 0) return;
  const int total = S * 24 * 256;
  dim3 grid((total + kSplitThreads - 1) / kSplitThreads);
  dim3 block(kSplitThreads);
  qwen36_split_q_gate_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q_proj),
      reinterpret_cast<__nv_bfloat16*>(q_pre),
      reinterpret_cast<__nv_bfloat16*>(gate),
      S);
}

void qwen36_gdn_gating_bf16(
    const void* a,
    const void* b,
    const float* neg_exp_A_log,
    const float* dt_bias,
    void*       g_out,
    void*       beta_out,
    int S,
    int num_heads,
    cudaStream_t stream)
{
  if (S <= 0 || num_heads <= 0) return;
  const int total = S * num_heads;
  dim3 grid((total + kSplitThreads - 1) / kSplitThreads);
  dim3 block(kSplitThreads);
  qwen36_gdn_gating_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(a),
      reinterpret_cast<const __nv_bfloat16*>(b),
      neg_exp_A_log,
      dt_bias,
      reinterpret_cast<__nv_bfloat16*>(g_out),
      reinterpret_cast<__nv_bfloat16*>(beta_out),
      S, num_heads);
}

void qwen36_gdn_gating_strided_bf16(
    const void* a,
    const void* b,
    const float* neg_exp_A_log,
    const float* dt_bias,
    void*       g_out,
    void*       beta_out,
    int S,
    int num_heads,
    int a_stride,
    int b_stride,
    cudaStream_t stream)
{
  if (S <= 0 || num_heads <= 0) return;
  const int total = S * num_heads;
  dim3 grid((total + kSplitThreads - 1) / kSplitThreads);
  dim3 block(kSplitThreads);
  qwen36_gdn_gating_strided_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(a),
      reinterpret_cast<const __nv_bfloat16*>(b),
      neg_exp_A_log,
      dt_bias,
      reinterpret_cast<__nv_bfloat16*>(g_out),
      reinterpret_cast<__nv_bfloat16*>(beta_out),
      S, num_heads, a_stride, b_stride);
}

void qwen36_gdn_chunk_from_conv_smem_bf16(
    const void* conv_out,
    const void* a,
    const void* b,
    const float* neg_exp_A_log,
    const float* dt_bias,
    void*       state,
    void*       out,
    int S,
    int num_v_heads,
    bool use_qk_l2norm,
    cudaStream_t stream)
{
  gdn_chunk_from_conv_smem_h_bf16(
      conv_out, a, b, neg_exp_A_log, dt_bias, state, out,
      S, num_v_heads, 16, kHD, num_v_heads, num_v_heads,
      use_qk_l2norm, stream);
}

void gdn_chunk_from_conv_smem_h_bf16(
    const void* conv_out,
    const void* a,
    const void* b,
    const float* neg_exp_A_log,
    const float* dt_bias,
    void*       state,
    void*       out,
    int S,
    int num_v_heads,
    int num_k_heads,
    int head_dim,
    int a_stride,
    int b_stride,
    bool use_qk_l2norm,
    cudaStream_t stream)
{
  if (S <= 0 || num_v_heads <= 0) return;
  if (head_dim != kHD) return;
  dim3 grid(num_v_heads, 1);
  dim3 block(kHD);
  constexpr size_t kSmemBytes =
      (kHD * kHD + 2 * kHD + 34) * sizeof(float);
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(
        qwen36_gdn_chunk_from_conv_smem_kernel<kHD>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kSmemBytes));
    attr_set = true;
  }
  qwen36_gdn_chunk_from_conv_smem_kernel<kHD><<<
      grid, block, kSmemBytes, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(conv_out),
      reinterpret_cast<const __nv_bfloat16*>(a),
      reinterpret_cast<const __nv_bfloat16*>(b),
      neg_exp_A_log,
      dt_bias,
      reinterpret_cast<__nv_bfloat16*>(state),
      reinterpret_cast<__nv_bfloat16*>(out),
      S, num_v_heads, num_k_heads, a_stride, b_stride, use_qk_l2norm);
}

void qwen36_gdn_chunk_from_conv_smem_strided_bf16(
    const void* conv_out,
    const void* a,
    const void* b,
    const float* neg_exp_A_log,
    const float* dt_bias,
    void*       state,
    void*       out,
    int S,
    int num_v_heads,
    int a_stride,
    int b_stride,
    bool use_qk_l2norm,
    cudaStream_t stream)
{
  if (S <= 0 || num_v_heads <= 0) return;
  dim3 grid(num_v_heads, 1);
  dim3 block(kHD);
  constexpr size_t kSmemBytes =
      (kHD * kHD + 2 * kHD + 34) * sizeof(float);
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(
        qwen36_gdn_chunk_from_conv_smem_kernel<kHD>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kSmemBytes));
    attr_set = true;
  }
  qwen36_gdn_chunk_from_conv_smem_kernel<kHD><<<
      grid, block, kSmemBytes, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(conv_out),
      reinterpret_cast<const __nv_bfloat16*>(a),
      reinterpret_cast<const __nv_bfloat16*>(b),
      neg_exp_A_log,
      dt_bias,
      reinterpret_cast<__nv_bfloat16*>(state),
      reinterpret_cast<__nv_bfloat16*>(out),
      S, num_v_heads, 16, a_stride, b_stride, use_qk_l2norm);
}

void gated_deltanet_chunk_smem_qwen36_bf16(
    const void* q,
    const void* k,
    const void* v,
    const void* g,
    const void* beta,
    void*       state,
    void*       out,
    int S, int num_v_heads, int head_k_dim, int head_v_dim,
    bool use_qk_l2norm,
    cudaStream_t stream)
{
  if (head_k_dim != kHD || head_v_dim != kHD || S <= 0) {
    return;
  }
  dim3 grid(num_v_heads, 1);
  dim3 block(kHD);
  constexpr size_t kSmemBytes =
      (kHD * kHD + 2 * kHD + 32) * sizeof(float);
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(
        gated_deltanet_chunk_smem_kernel<kHD>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kSmemBytes));
    attr_set = true;
  }
  gated_deltanet_chunk_smem_kernel<kHD><<<
      grid, block, kSmemBytes, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q),
      reinterpret_cast<const __nv_bfloat16*>(k),
      reinterpret_cast<const __nv_bfloat16*>(v),
      reinterpret_cast<const __nv_bfloat16*>(g),
      reinterpret_cast<const __nv_bfloat16*>(beta),
      reinterpret_cast<__nv_bfloat16*>(state),
      reinterpret_cast<__nv_bfloat16*>(out),
      S, num_v_heads, use_qk_l2norm);
}

void qwen36_gdn_wy_norm_cumsum_bf16(
    const void* q16,
    const void* k16,
    const void* g,
    void*       q16_l2,
    void*       k16_l2,
    void*       g_cumsum,
    int S,
    cudaStream_t stream)
{
  if (S <= 0) return;
  qwen36_gdn_wy_norm_qk_kernel<<<dim3(kQHeads, S), kHD, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q16),
      reinterpret_cast<const __nv_bfloat16*>(k16),
      reinterpret_cast<__nv_bfloat16*>(q16_l2),
      reinterpret_cast<__nv_bfloat16*>(k16_l2),
      nullptr,
      nullptr,
      S, kQHeads, kVHeads, kVHeads / kQHeads);
  qwen36_gdn_wy_cumsum_g_kernel<<<1, 64, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(g),
      reinterpret_cast<__nv_bfloat16*>(g_cumsum),
      S, kVHeads);
}

void qwen36_gdn_wy_norm_cumsum_pack_q_bf16(
    const void* q16,
    const void* k16,
    const void* g,
    void*       q16_l2,
    void*       k16_l2,
    void*       q_pack_hv,
    void*       g_cumsum,
    int S,
    cudaStream_t stream)
{
  if (S <= 0) return;
  qwen36_gdn_wy_norm_qk_kernel<<<dim3(kQHeads, S), kHD, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q16),
      reinterpret_cast<const __nv_bfloat16*>(k16),
      reinterpret_cast<__nv_bfloat16*>(q16_l2),
      reinterpret_cast<__nv_bfloat16*>(k16_l2),
      reinterpret_cast<__nv_bfloat16*>(q_pack_hv),
      nullptr,
      S, kQHeads, kVHeads, kVHeads / kQHeads);
  qwen36_gdn_wy_cumsum_g_kernel<<<1, 64, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(g),
      reinterpret_cast<__nv_bfloat16*>(g_cumsum),
      S, kVHeads);
}

void qwen36_gdn_wy_norm_cumsum_pack_qk_bf16(
    const void* q16,
    const void* k16,
    const void* g,
    void*       q16_l2,
    void*       k16_l2,
    void*       q_pack_hv,
    void*       k_pack_hk,
    void*       g_cumsum,
    int S,
    cudaStream_t stream)
{
  if (S <= 0) return;
  qwen36_gdn_wy_norm_qk_kernel<<<dim3(kQHeads, S), kHD, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q16),
      reinterpret_cast<const __nv_bfloat16*>(k16),
      reinterpret_cast<__nv_bfloat16*>(q16_l2),
      reinterpret_cast<__nv_bfloat16*>(k16_l2),
      reinterpret_cast<__nv_bfloat16*>(q_pack_hv),
      reinterpret_cast<__nv_bfloat16*>(k_pack_hk),
      S, kQHeads, kVHeads, kVHeads / kQHeads);
  qwen36_gdn_wy_cumsum_g_kernel<<<1, 64, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(g),
      reinterpret_cast<__nv_bfloat16*>(g_cumsum),
      S, kVHeads);
}

void qwen36_gdn_wy_kkt_b64_bf16(
    const void* k16_l2,
    const void* beta,
    const void* g_cumsum,
    void*       A,
    int S,
    cudaStream_t stream)
{
  if (S <= 0) return;
  const int chunks = (S + kWyChunk - 1) / kWyChunk;
  const int pairs = kWyChunk * kWyChunk;
  qwen36_gdn_wy_kkt_b64_kernel<<<
      dim3((pairs + 255) / 256, kVHeads, chunks), 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(k16_l2),
      reinterpret_cast<const __nv_bfloat16*>(beta),
      reinterpret_cast<const __nv_bfloat16*>(g_cumsum),
      reinterpret_cast<float*>(A),
      S, kQHeads, kVHeads, kVHeads / kQHeads);
}

void qwen36_gdn_wy_solve_tril_b64_f32(
    const void* A,
    void*       Ai,
    int S,
    cudaStream_t stream)
{
  if (S <= 0) return;
  const int chunks = (S + kWyChunk - 1) / kWyChunk;
  qwen36_gdn_wy_solve_tril_b64_kernel<<<
      dim3(kVHeads, chunks), 64, 0, stream>>>(
      reinterpret_cast<const float*>(A),
      reinterpret_cast<float*>(Ai),
      S, kVHeads);
}

void qwen36_gdn_wy_cast_ai_f32_to_bf16(
    const void* Ai,
    void*       Ai_pack,
    int S,
    cudaStream_t stream)
{
  if (S <= 0) return;
  const int chunks = (S + kWyChunk - 1) / kWyChunk;
  const int total = chunks * kVHeads * kWyChunk * kWyChunk;
  qwen36_gdn_wy_cast_ai_f32_to_bf16_kernel<<<
      (total + 255) / 256, 256, 0, stream>>>(
      reinterpret_cast<const float*>(Ai),
      reinterpret_cast<__nv_bfloat16*>(Ai_pack),
      total);
}

void qwen36_gdn_wy_recompute_wu_b64_bf16(
    const void* k16_l2,
    const void* v48,
    const void* beta,
    const void* g_cumsum,
    const void* Ai,
    void*       w48,
    void*       u48,
    int S,
    cudaStream_t stream)
{
  if (S <= 0) return;
  const int total = S * kVHeads * kHD;
  qwen36_gdn_wy_recompute_wu_b64_kernel<<<
      (total + 255) / 256, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(k16_l2),
      reinterpret_cast<const __nv_bfloat16*>(v48),
      reinterpret_cast<const __nv_bfloat16*>(beta),
      reinterpret_cast<const __nv_bfloat16*>(g_cumsum),
      reinterpret_cast<const float*>(Ai),
      reinterpret_cast<__nv_bfloat16*>(w48),
      reinterpret_cast<__nv_bfloat16*>(u48),
      S, kQHeads, kVHeads, kVHeads / kQHeads);
}

void qwen36_gdn_wy_chunk_h_b64_bf16(
    const void* k16_l2,
    const void* u48,
    const void* w48,
    const void* g_cumsum,
    void*       state,
    void*       h0,
    void*       v_new,
    int S,
    cudaStream_t stream)
{
  if (S <= 0) return;
  constexpr size_t kSmemBytes = kHD * kHD * sizeof(float);
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(
        qwen36_gdn_wy_chunk_h_b64_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kSmemBytes));
    attr_set = true;
  }
  qwen36_gdn_wy_chunk_h_b64_kernel<<<
      kVHeads, kHD, kSmemBytes, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(k16_l2),
      reinterpret_cast<const __nv_bfloat16*>(u48),
      reinterpret_cast<const __nv_bfloat16*>(w48),
      reinterpret_cast<const __nv_bfloat16*>(g_cumsum),
      reinterpret_cast<__nv_bfloat16*>(state),
      reinterpret_cast<__nv_bfloat16*>(h0),
      reinterpret_cast<__nv_bfloat16*>(v_new),
      S, kQHeads, kVHeads, kVHeads / kQHeads);
}

void qwen36_gdn_wy_output_o_b64_bf16(
    const void* q16_l2,
    const void* k16_l2,
    const void* v_new,
    const void* h0,
    const void* g_cumsum,
    void*       out,
    int S,
    cudaStream_t stream)
{
  if (S <= 0) return;
  const int total = S * kVHeads * kHD;
  qwen36_gdn_wy_output_o_b64_kernel<<<
      (total + 255) / 256, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q16_l2),
      reinterpret_cast<const __nv_bfloat16*>(k16_l2),
      reinterpret_cast<const __nv_bfloat16*>(v_new),
      reinterpret_cast<const __nv_bfloat16*>(h0),
      reinterpret_cast<const __nv_bfloat16*>(g_cumsum),
      reinterpret_cast<__nv_bfloat16*>(out),
      S, kQHeads, kVHeads, kVHeads / kQHeads);
}

void gdn_wy_norm_cumsum_pack_qk_h_bf16(
    const void* q, const void* k, const void* g, void* q_l2, void* k_l2,
    void* q_pack_hv, void* k_pack_hk, void* g_cumsum, int S,
    int num_v_heads, int num_k_heads, int head_dim, cudaStream_t stream) {
  if (S <= 0) return;
  const int head_group_size = num_v_heads / num_k_heads;
  qwen36_gdn_wy_norm_qk_kernel<<<dim3(num_k_heads, S), head_dim, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q),
      reinterpret_cast<const __nv_bfloat16*>(k),
      reinterpret_cast<__nv_bfloat16*>(q_l2),
      reinterpret_cast<__nv_bfloat16*>(k_l2),
      reinterpret_cast<__nv_bfloat16*>(q_pack_hv),
      reinterpret_cast<__nv_bfloat16*>(k_pack_hk), S, num_k_heads,
      num_v_heads, head_group_size);
  qwen36_gdn_wy_cumsum_g_kernel<<<
      (num_v_heads + 63) / 64, 64, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(g),
      reinterpret_cast<__nv_bfloat16*>(g_cumsum), S, num_v_heads);
}

void gdn_wy_kkt_b64_h_bf16(
    const void* k_l2, const void* beta, const void* g_cumsum, void* A,
    int S, int num_v_heads, int num_k_heads, int head_dim,
    cudaStream_t stream) {
  if (S <= 0) return;
  const int chunks = (S + kWyChunk - 1) / kWyChunk;
  const int pairs = kWyChunk * kWyChunk;
  qwen36_gdn_wy_kkt_b64_kernel<<<
      dim3((pairs + 255) / 256, num_v_heads, chunks), 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(k_l2),
      reinterpret_cast<const __nv_bfloat16*>(beta),
      reinterpret_cast<const __nv_bfloat16*>(g_cumsum),
      reinterpret_cast<float*>(A), S, num_k_heads, num_v_heads,
      num_v_heads / num_k_heads);
}

void gdn_wy_solve_tril_b64_h_f32(
    const void* A, void* Ai, int S, int num_v_heads, cudaStream_t stream) {
  if (S <= 0) return;
  const int chunks = (S + kWyChunk - 1) / kWyChunk;
  qwen36_gdn_wy_solve_tril_b64_kernel<<<
      dim3(num_v_heads, chunks), 64, 0, stream>>>(
      reinterpret_cast<const float*>(A), reinterpret_cast<float*>(Ai), S,
      num_v_heads);
}

void gdn_wy_cast_ai_h_f32_to_bf16(
    const void* Ai, void* Ai_pack, int S, int num_v_heads,
    cudaStream_t stream) {
  if (S <= 0) return;
  const int chunks = (S + kWyChunk - 1) / kWyChunk;
  const int total = chunks * num_v_heads * kWyChunk * kWyChunk;
  qwen36_gdn_wy_cast_ai_f32_to_bf16_kernel<<<
      (total + 255) / 256, 256, 0, stream>>>(
      reinterpret_cast<const float*>(Ai),
      reinterpret_cast<__nv_bfloat16*>(Ai_pack), total);
}

void gdn_wy_recompute_wu_b64_h_bf16(
    const void* k_l2, const void* v, const void* beta,
    const void* g_cumsum, const void* Ai, void* w, void* u, int S,
    int num_v_heads, int num_k_heads, int head_dim, cudaStream_t stream) {
  if (S <= 0) return;
  const int total = S * num_v_heads * head_dim;
  qwen36_gdn_wy_recompute_wu_b64_kernel<<<
      (total + 255) / 256, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(k_l2),
      reinterpret_cast<const __nv_bfloat16*>(v),
      reinterpret_cast<const __nv_bfloat16*>(beta),
      reinterpret_cast<const __nv_bfloat16*>(g_cumsum),
      reinterpret_cast<const float*>(Ai),
      reinterpret_cast<__nv_bfloat16*>(w),
      reinterpret_cast<__nv_bfloat16*>(u), S, num_k_heads, num_v_heads,
      num_v_heads / num_k_heads);
}

void gdn_wy_chunk_h_b64_h_bf16(
    const void* k_l2, const void* u, const void* w,
    const void* g_cumsum, void* state, void* h0, void* v_new, int S,
    int num_v_heads, int num_k_heads, int head_dim, cudaStream_t stream) {
  if (S <= 0) return;
  constexpr size_t kSmemBytes = kHD * kHD * sizeof(float);
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(qwen36_gdn_wy_chunk_h_b64_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(kSmemBytes));
    attr_set = true;
  }
  qwen36_gdn_wy_chunk_h_b64_kernel<<<
      num_v_heads, head_dim, kSmemBytes, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(k_l2),
      reinterpret_cast<const __nv_bfloat16*>(u),
      reinterpret_cast<const __nv_bfloat16*>(w),
      reinterpret_cast<const __nv_bfloat16*>(g_cumsum),
      reinterpret_cast<__nv_bfloat16*>(state),
      reinterpret_cast<__nv_bfloat16*>(h0),
      reinterpret_cast<__nv_bfloat16*>(v_new), S, num_k_heads, num_v_heads,
      num_v_heads / num_k_heads);
}

void gdn_wy_output_o_b64_h_bf16(
    const void* q_l2, const void* k_l2, const void* v_new,
    const void* h0, const void* g_cumsum, void* out, int S,
    int num_v_heads, int num_k_heads, int head_dim, cudaStream_t stream) {
  if (S <= 0) return;
  const int total = S * num_v_heads * head_dim;
  qwen36_gdn_wy_output_o_b64_kernel<<<
      (total + 255) / 256, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q_l2),
      reinterpret_cast<const __nv_bfloat16*>(k_l2),
      reinterpret_cast<const __nv_bfloat16*>(v_new),
      reinterpret_cast<const __nv_bfloat16*>(h0),
      reinterpret_cast<const __nv_bfloat16*>(g_cumsum),
      reinterpret_cast<__nv_bfloat16*>(out), S, num_k_heads, num_v_heads,
      num_v_heads / num_k_heads);
}

}  // namespace kernels
}  // namespace flash_rt
