// SPDX-License-Identifier: Apache-2.0
// Per-token AdaLayerNorm + per-tensor FP8 quantize.
//
// Same math as ada_layer_norm_fp8.cu::ada_layer_norm_fp8_kernel with one
// difference: the modulation vectors are per token ([M, D], one row per
// activation row) instead of broadcast per channel ([D]). This is the
// producer form modern video DiTs need — their timestep embedding carries
// a scale/shift *per token* (e.g. a [B, S, 6, D] table chunked per site),
// so a [D] broadcast producer cannot serve the seam at all.

#include "ada_layer_norm_fp8.cuh"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace quantize {

namespace {

constexpr float kFp8MaxPtok = 448.0f;

__global__ void ada_layer_norm_ptok_fp8_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ shift,
    __nv_fp8_e4m3*       __restrict__ out,
    const float*          __restrict__ act_scale_ptr,
    int dim, float eps)
{
  const int row = blockIdx.x;
  const __nv_bfloat162* x2 =
      reinterpret_cast<const __nv_bfloat162*>(x + (long long)row * dim);
  const __nv_bfloat162* sc2 =
      reinterpret_cast<const __nv_bfloat162*>(scale + (long long)row * dim);
  const __nv_bfloat162* sh2 =
      reinterpret_cast<const __nv_bfloat162*>(shift + (long long)row * dim);
  __nv_fp8_e4m3* out_row = out + (long long)row * dim;
  const int dim2 = dim >> 1;

  extern __shared__ float shared[];

  // ── Pass 1: mean ──
  float local_sum = 0.f;
  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    __nv_bfloat162 v = x2[i];
    local_sum += __bfloat162float(v.x) + __bfloat162float(v.y);
  }
  float val = local_sum;
  for (int o = 16; o > 0; o >>= 1)
    val += __shfl_xor_sync(0xffffffffu, val, o);
  const int lane = threadIdx.x & 31;
  const int wid  = threadIdx.x >> 5;
  if (!lane) shared[wid] = val;
  __syncthreads();
  if (!wid) {
    val = (lane < (blockDim.x >> 5)) ? shared[lane] : 0.f;
    for (int o = 16; o > 0; o >>= 1)
      val += __shfl_xor_sync(0xffffffffu, val, o);
  }
  __syncthreads();
  if (!threadIdx.x) shared[0] = val;
  __syncthreads();
  const float mean = shared[0] / static_cast<float>(dim);

  // ── Pass 2: variance ──
  float local_var = 0.f;
  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    __nv_bfloat162 v = x2[i];
    const float d0 = __bfloat162float(v.x) - mean;
    const float d1 = __bfloat162float(v.y) - mean;
    local_var += d0 * d0 + d1 * d1;
  }
  val = local_var;
  for (int o = 16; o > 0; o >>= 1)
    val += __shfl_xor_sync(0xffffffffu, val, o);
  if (!lane) shared[wid] = val;
  __syncthreads();
  if (!wid) {
    val = (lane < (blockDim.x >> 5)) ? shared[lane] : 0.f;
    for (int o = 16; o > 0; o >>= 1)
      val += __shfl_xor_sync(0xffffffffu, val, o);
  }
  __syncthreads();
  if (!threadIdx.x) shared[0] = val;
  __syncthreads();
  const float inv_std = rsqrtf(shared[0] / static_cast<float>(dim) + eps);

  // ── Pass 3: per-token ada_modulate + fp8 quantize ──
  // Register-level round-through bf16 before fp8, matching the family's
  // reference chain semantics (see ada_layer_norm_fp8.cu).
  const float inv_a = 1.0f / *act_scale_ptr;
  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    __nv_bfloat162 xv = x2[i], sv = sc2[i], hv = sh2[i];
    const float n0 = (__bfloat162float(xv.x) - mean) * inv_std;
    const float n1 = (__bfloat162float(xv.y) - mean) * inv_std;
    const float v0_f32 = n0 * (1.0f + __bfloat162float(sv.x))
                          + __bfloat162float(hv.x);
    const float v1_f32 = n1 * (1.0f + __bfloat162float(sv.y))
                          + __bfloat162float(hv.y);
    const float v0 = __bfloat162float(__float2bfloat16(v0_f32));
    const float v1 = __bfloat162float(__float2bfloat16(v1_f32));
    float q0 = fminf(fmaxf(v0 * inv_a, -kFp8MaxPtok), kFp8MaxPtok);
    float q1 = fminf(fmaxf(v1 * inv_a, -kFp8MaxPtok), kFp8MaxPtok);
    out_row[2 * i]     = __nv_fp8_e4m3(q0);
    out_row[2 * i + 1] = __nv_fp8_e4m3(q1);
  }
}

}  // namespace

void ada_layer_norm_ptok_fp8(
    const void*  x_bf16,
    const void*  scale_bf16,
    const void*  shift_bf16,
    void*        out_fp8,
    const float* act_scale,
    int seq_len, int dim, float eps,
    cudaStream_t stream)
{
  if (seq_len <= 0 || dim <= 0) return;
  ada_layer_norm_ptok_fp8_kernel<<<seq_len, 256, 256 * sizeof(float),
                                   stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x_bf16),
      reinterpret_cast<const __nv_bfloat16*>(scale_bf16),
      reinterpret_cast<const __nv_bfloat16*>(shift_bf16),
      reinterpret_cast<__nv_fp8_e4m3*>(out_fp8),
      act_scale,
      dim, eps);
}

}  // namespace quantize
}  // namespace flash_rt

namespace flash_rt {
namespace quantize {

namespace {

__global__ void ada_layer_norm_ptok_table_fp8_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ temb,   // [M, n_chunks, D]
    const float*         __restrict__ table,  // [n_chunks, D]
    __nv_fp8_e4m3*       __restrict__ out,
    __nv_bfloat16*       __restrict__ out_bf16,
    const float*          __restrict__ act_scale_ptr,
    int dim, int n_chunks, int shift_idx, int scale_idx, float eps)
{
  const int row = blockIdx.x;
  const __nv_bfloat162* x2 =
      reinterpret_cast<const __nv_bfloat162*>(x + (long long)row * dim);
  const __nv_bfloat162* sc2 = reinterpret_cast<const __nv_bfloat162*>(
      temb + ((long long)row * n_chunks + scale_idx) * dim);
  const __nv_bfloat162* sh2 = reinterpret_cast<const __nv_bfloat162*>(
      temb + ((long long)row * n_chunks + shift_idx) * dim);
  const float2* tsc2 = reinterpret_cast<const float2*>(
      table + (long long)scale_idx * dim);
  const float2* tsh2 = reinterpret_cast<const float2*>(
      table + (long long)shift_idx * dim);
  __nv_fp8_e4m3* out_row = out ? out + (long long)row * dim : nullptr;
  __nv_bfloat16* out_bf16_row = out_bf16 ? out_bf16 + (long long)row * dim : nullptr;
  const int dim2 = dim >> 1;

  extern __shared__ float shared[];

  float local_sum = 0.f;
  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    __nv_bfloat162 v = x2[i];
    local_sum += __bfloat162float(v.x) + __bfloat162float(v.y);
  }
  float val = local_sum;
  for (int o = 16; o > 0; o >>= 1)
    val += __shfl_xor_sync(0xffffffffu, val, o);
  const int lane = threadIdx.x & 31;
  const int wid  = threadIdx.x >> 5;
  if (!lane) shared[wid] = val;
  __syncthreads();
  if (!wid) {
    val = (lane < (blockDim.x >> 5)) ? shared[lane] : 0.f;
    for (int o = 16; o > 0; o >>= 1)
      val += __shfl_xor_sync(0xffffffffu, val, o);
  }
  __syncthreads();
  if (!threadIdx.x) shared[0] = val;
  __syncthreads();
  const float mean = shared[0] / static_cast<float>(dim);

  float local_var = 0.f;
  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    __nv_bfloat162 v = x2[i];
    const float d0 = __bfloat162float(v.x) - mean;
    const float d1 = __bfloat162float(v.y) - mean;
    local_var += d0 * d0 + d1 * d1;
  }
  val = local_var;
  for (int o = 16; o > 0; o >>= 1)
    val += __shfl_xor_sync(0xffffffffu, val, o);
  if (!lane) shared[wid] = val;
  __syncthreads();
  if (!wid) {
    val = (lane < (blockDim.x >> 5)) ? shared[lane] : 0.f;
    for (int o = 16; o > 0; o >>= 1)
      val += __shfl_xor_sync(0xffffffffu, val, o);
  }
  __syncthreads();
  if (!threadIdx.x) shared[0] = val;
  __syncthreads();
  const float inv_std = rsqrtf(shared[0] / static_cast<float>(dim) + eps);

  const float inv_a = act_scale_ptr ? 1.0f / *act_scale_ptr : 0.0f;
  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    __nv_bfloat162 xv = x2[i], sv = sc2[i], hv = sh2[i];
    const float2 ts = tsc2[i], th = tsh2[i];
    const float s0 = ts.x + __bfloat162float(sv.x);
    const float s1 = ts.y + __bfloat162float(sv.y);
    const float h0 = th.x + __bfloat162float(hv.x);
    const float h1 = th.y + __bfloat162float(hv.y);
    const float n0 = (__bfloat162float(xv.x) - mean) * inv_std;
    const float n1 = (__bfloat162float(xv.y) - mean) * inv_std;
    const float v0 = __bfloat162float(
        __float2bfloat16(n0 * (1.0f + s0) + h0));
    const float v1 = __bfloat162float(
        __float2bfloat16(n1 * (1.0f + s1) + h1));
    if (out_bf16_row) {
      reinterpret_cast<__nv_bfloat162*>(out_bf16_row)[i] =
          __floats2bfloat162_rn(v0, v1);
    } else {
      float q0 = fminf(fmaxf(v0 * inv_a, -kFp8MaxPtok), kFp8MaxPtok);
      float q1 = fminf(fmaxf(v1 * inv_a, -kFp8MaxPtok), kFp8MaxPtok);
      out_row[2 * i]     = __nv_fp8_e4m3(q0);
      out_row[2 * i + 1] = __nv_fp8_e4m3(q1);
    }
  }
}

}  // namespace

void ada_layer_norm_ptok_table_fp8(
    const void*  x_bf16,
    const void*  temb_bf16,
    const float* table_f32,
    void*        out_fp8,
    const float* act_scale,
    int seq_len, int dim, int n_chunks, int shift_idx, int scale_idx,
    float eps, cudaStream_t stream)
{
  if (seq_len <= 0 || dim <= 0) return;
  ada_layer_norm_ptok_table_fp8_kernel<<<seq_len, 256,
                                         256 * sizeof(float), stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x_bf16),
      reinterpret_cast<const __nv_bfloat16*>(temb_bf16),
      table_f32,
      reinterpret_cast<__nv_fp8_e4m3*>(out_fp8),
      nullptr,
      act_scale,
      dim, n_chunks, shift_idx, scale_idx, eps);
}

void ada_layer_norm_ptok_table_bf16(
    const void* x_bf16, const void* temb_bf16, const float* table_f32,
    void* out_bf16, int seq_len, int dim, int n_chunks, int shift_idx,
    int scale_idx, float eps, cudaStream_t stream)
{
  if (seq_len <= 0 || dim <= 0) return;
  ada_layer_norm_ptok_table_fp8_kernel<<<seq_len, 256,
                                         256 * sizeof(float), stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x_bf16),
      reinterpret_cast<const __nv_bfloat16*>(temb_bf16), table_f32, nullptr,
      reinterpret_cast<__nv_bfloat16*>(out_bf16), nullptr, dim, n_chunks,
      shift_idx, scale_idx, eps);
}

}  // namespace quantize
}  // namespace flash_rt
