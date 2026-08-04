// SPDX-License-Identifier: Apache-2.0
//
// Portable SIMT recurrent sequence scan for the Gated DeltaNet prefill path.
//
// Replicates the per-token update of gated_deltanet_recurrent_qwen36_bf16
// (q/k L2 norm when use_qk_l2norm, q scaled by 1/sqrt(HD), g = exp(log-decay),
// delta-rule state update) but scans the whole sequence in one launch with
// the state held in FP32 registers across all timesteps, and writes the
// state to BF16 once at the end -- the same semantics as the sm_120a
// gdn_recurrent_seq kernel. sm_120 keeps the MMA path; this is a pure SIMT
// compatibility path for sm_110 Thor.
//
// Layout (all bf16, single batch):
//   q/k/v/out : (S, num_v_heads, HD)
//   g/beta    : (S, num_v_heads)
//   state     : (num_v_heads, HD, HD)  -- read as initial, overwritten final
// HD = 128.

#include "gdn_recurrent_seq_simt.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace kernels {

namespace {

constexpr int kHD = 128;
constexpr float kEps = 1e-6f;

template <int HD>
__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
  for (int off = 16; off > 0; off >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, off);
  }
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

__global__ void gdn_recurrent_seq_simt_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ g,
    const __nv_bfloat16* __restrict__ beta,
    __nv_bfloat16* __restrict__ state,
    __nv_bfloat16* __restrict__ out,
    int S, int H, bool use_qk_l2norm) {
  const int h = blockIdx.x;
  const int t = threadIdx.x;
  if (t >= kHD) return;

  __shared__ float smem[2 * kHD + 32];
  __shared__ float ksh[kHD];
  __shared__ float qsh[kHD];
  float* scratch = smem + 2 * kHD;

  // FP32 state column in registers: col[i] = state[h, i, t].
  float col[kHD];
  const size_t state_off = (size_t)h * kHD * kHD;
#pragma unroll 16
  for (int i = 0; i < kHD; ++i) {
    col[i] = static_cast<float>(state[state_off + (size_t)i * kHD + t]);
  }

  for (int s = 0; s < S; ++s) {
    const size_t qoff = ((size_t)s * H + h) * kHD + t;
    float qs = static_cast<float>(q[qoff]);
    float ks = static_cast<float>(k[qoff]);

    if (use_qk_l2norm) {
      float q_sq = qs * qs;
      float k_sq = ks * ks;
      q_sq = block_reduce_sum<kHD>(q_sq, scratch);
      k_sq = block_reduce_sum<kHD>(k_sq, scratch);
      qs *= rsqrtf(q_sq + kEps);
      ks *= rsqrtf(k_sq + kEps);
    }
    qs *= rsqrtf(static_cast<float>(kHD));

    // Publish the normalized per-element k/q so every thread can contract the
    // full HD dimension: kv_mem[t] = sum_i col[i]*ks[i], out[t] = sum_i
    // col[i]*qs[i] (col[i] = state[h, i, t] is already in registers).
    ksh[t] = ks;
    qsh[t] = qs;
    __syncthreads();

    const float g_t = __expf(static_cast<float>(g[s * H + h]));
    const float beta_t = static_cast<float>(beta[s * H + h]);

    // Decay: st[i, t] *= exp(g)
#pragma unroll 16
    for (int i = 0; i < kHD; ++i) {
      col[i] *= g_t;
    }

    // kv_mem = sum_i col[i] * ks[i]
    float kv_mem = 0.0f;
#pragma unroll 16
    for (int i = 0; i < kHD; ++i) {
      kv_mem += col[i] * ksh[i];
    }

    // delta = (v[t] - kv_mem) * beta ;  col[i] += ks[i] * delta
    const float v_t =
        static_cast<float>(v[((size_t)s * H + h) * kHD + t]);
    const float delta = (v_t - kv_mem) * beta_t;
#pragma unroll 16
    for (int i = 0; i < kHD; ++i) {
      col[i] = fmaf(ksh[i], delta, col[i]);
    }

    // out[s, h, t] = sum_i col[i] * qs[i]
    float out_t = 0.0f;
#pragma unroll 16
    for (int i = 0; i < kHD; ++i) {
      out_t += col[i] * qsh[i];
    }
    out[qoff] = __float2bfloat16(out_t);
    __syncthreads();  // before the next iteration overwrites ksh/qsh
  }

#pragma unroll 16
  for (int i = 0; i < kHD; ++i) {
    state[state_off + (size_t)i * kHD + t] = __float2bfloat16(col[i]);
  }
}

}  // namespace

int gdn_recurrent_seq_bf16_simt(
    const void* q, const void* k, const void* v, const void* g,
    const void* beta, void* state, void* out, int S, int num_v_heads,
    int head_dim, bool use_qk_l2norm, cudaStream_t stream) {
  if (head_dim != kHD) return 1;  // Qwen3.6 profile requires HD=128
  if (S <= 0 || num_v_heads <= 0) return 1;
  dim3 grid(num_v_heads);
  dim3 block(kHD);
  gdn_recurrent_seq_simt_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q),
      reinterpret_cast<const __nv_bfloat16*>(k),
      reinterpret_cast<const __nv_bfloat16*>(v),
      reinterpret_cast<const __nv_bfloat16*>(g),
      reinterpret_cast<const __nv_bfloat16*>(beta),
      reinterpret_cast<__nv_bfloat16*>(state),
      reinterpret_cast<__nv_bfloat16*>(out), S, num_v_heads, use_qk_l2norm);
  return (cudaGetLastError() == cudaSuccess) ? 0 : 1;
}

}  // namespace kernels
}  // namespace flash_rt
