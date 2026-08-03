// SPDX-License-Identifier: Apache-2.0
//
// Portable SIMT reference implementations of the world-model-conv kernels.
//
// The sm_120a kernels use the `mma.sync.aligned.kind::f8f6f4` /
// `mxf4nvf4.block_scale` tensor core instructions that do not exist on
// earlier architectures (e.g. sm_110 Thor). These reference kernels compute
// the same convolutions in pure SIMT FMA so the package is usable (slowly)
// on pre-sm_120 devices. sm_120 keeps the tensor-core path.
//
// Semantics (causal, SAME-padded 3x3/3x3x3 conv, FP8 e4m3 or NVFP4 e2m1 +
// ue4m3 per-16-channel scale):
//   fp8:  out = residual + alpha * sum_k x[win]*w[co,win] + bias[co]
//   fp4:  a[k] = fp4(x[k]) * ue4m3(sf_x[k/16]), b[k] = fp4(w[k])*ue4m3(sf_w[k/16])
//         out = residual + alpha * sum_k a[k]*b[k] + bias[co]  (v2: *outer[co])

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace flash_rt {
namespace conv {

namespace {

constexpr int THREADS = 256;

__device__ __forceinline__ float fp4_to_float(uint8_t v) {
  int sign = (v & 8) ? -1 : 1;
  int exp = (v >> 1) & 3;
  int man = v & 1;
  if (exp == 0 && man == 0) return sign * 0.0f;
  float val = (exp == 0) ? (0.5f * man)
                         : (float)(1 << (exp - 1)) * (1.0f + 0.5f * man);
  return sign * val;
}

__device__ __forceinline__ float ue4m3_to_float(uint8_t v) {
  int e = (v >> 3) & 0xF;
  int m = v & 7;
  if (e == 0) return m * (1.0f / 512.0f);
  return (1.0f + m * (1.0f / 8.0f)) * exp2f(static_cast<float>(e - 7));
}

__device__ __forceinline__ float fp4_read(const uint8_t* p, int idx) {
  uint8_t byte = p[idx >> 1];
  return fp4_to_float((idx & 1) ? (byte >> 4) : (byte & 0xF));
}

__device__ __forceinline__ float ue4_read(const uint8_t* p, int block) {
  return ue4m3_to_float(p[block]);
}

__device__ __forceinline__ int causal_src(const __nv_fp8_e4m3* cache_x,
                                          const __nv_fp8_e4m3* new_x,
                                          int b, int d_in,
                                          int h_in, int w_in,
                                          int T_cache, int T_new,
                                          int H, int W, int Ci) {
  if (h_in < 0 || h_in >= H || w_in < 0 || w_in >= W) return -1;
  if (d_in < T_cache) {
    return (((b * T_cache + d_in) * H + h_in) * W + w_in) * Ci;
  } else {
    int dn = d_in - T_cache;
    if (dn >= T_new) return -1;
    return (((b * T_new + dn) * H + h_in) * W + w_in) * Ci;
  }
}

// ---- FP8 causal conv3d, output NCDHW [N, Co, T_new, H, W] + residual ----
__global__ void fp8_conv3d_ncdhw_res_kernel(
    const __nv_fp8_e4m3* __restrict__ cache_x,
    const __nv_fp8_e4m3* __restrict__ new_x,
    const __nv_fp8_e4m3* __restrict__ w,
    const __nv_bfloat16* __restrict__ bias,
    const __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ y,
    int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
    float alpha) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * Co * T_new * H * W;
  if (idx >= total) return;
  int w_out = idx % W;
  int h = (idx / W) % H;
  int t = (idx / (W * H)) % T_new;
  int co = (idx / (W * H * T_new)) % Co;
  int b = idx / (W * H * T_new * Co);
  float acc = 0.0f;
  for (int kt = 0; kt < 3; ++kt)
    for (int kr = 0; kr < 3; ++kr)
      for (int ks = 0; ks < 3; ++ks) {
        int d = t + kt;
        int off = causal_src(cache_x, new_x, b, d, h + kr - 1, w_out + ks - 1,
                             T_cache, T_new, H, W, Ci);
        if (off < 0) continue;
        const __nv_fp8_e4m3* xs = (d < T_cache) ? cache_x + off : new_x + off;
        const __nv_fp8_e4m3* wr = w + ((((co * 3 + kt) * 3 + kr) * 3 + ks)) * Ci;
        for (int ci = 0; ci < Ci; ++ci) acc += float(xs[ci]) * float(wr[ci]);
      }
  float v = acc * alpha + __bfloat162float(bias[co]);
  v += __bfloat162float(residual[idx]);
  y[idx] = __float2bfloat16(v);
}

// ---- FP8 causal conv3d, output NDHWC [N, T_new, H, W, Co] ----
__global__ void fp8_conv3d_ndhwc_kernel(
    const __nv_fp8_e4m3* __restrict__ cache_x,
    const __nv_fp8_e4m3* __restrict__ new_x,
    const __nv_fp8_e4m3* __restrict__ w,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ y,
    int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
    float alpha) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * T_new * H * W * Co;
  if (idx >= total) return;
  int co = idx % Co;
  int w_out = (idx / Co) % W;
  int h = (idx / (Co * W)) % H;
  int t = (idx / (Co * W * H)) % T_new;
  int b = idx / (Co * W * H * T_new);
  float acc = 0.0f;
  for (int kt = 0; kt < 3; ++kt)
    for (int kr = 0; kr < 3; ++kr)
      for (int ks = 0; ks < 3; ++ks) {
        int d = t + kt;
        int off = causal_src(cache_x, new_x, b, d, h + kr - 1, w_out + ks - 1,
                             T_cache, T_new, H, W, Ci);
        if (off < 0) continue;
        const __nv_fp8_e4m3* xs = (d < T_cache) ? cache_x + off : new_x + off;
        const __nv_fp8_e4m3* wr = w + ((((co * 3 + kt) * 3 + kr) * 3 + ks)) * Ci;
        for (int ci = 0; ci < Ci; ++ci) acc += float(xs[ci]) * float(wr[ci]);
      }
  float v = acc * alpha + __bfloat162float(bias[co]);
  y[idx] = __float2bfloat16(v);
}

// ---- FP8 2D 3x3 SAME conv, output NHWC [N, H, W, Co] ----
__global__ void fp8_conv2d_nhwc_kernel(
    const __nv_fp8_e4m3* __restrict__ x,
    const __nv_fp8_e4m3* __restrict__ w,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ y,
    int N, int H, int W, int Ci, int Co, float alpha) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * H * W * Co;
  if (idx >= total) return;
  int co = idx % Co;
  int w_out = (idx / Co) % W;
  int h = (idx / (Co * W)) % H;
  int n = idx / (Co * W * H);
  float acc = 0.0f;
  for (int kr = 0; kr < 3; ++kr)
    for (int ks = 0; ks < 3; ++ks) {
      int hi = h + kr - 1, wi = w_out + ks - 1;
      if (hi < 0 || hi >= H || wi < 0 || wi >= W) continue;
      const __nv_fp8_e4m3* xs = x + (((n * H + hi) * W + wi)) * Ci;
      const __nv_fp8_e4m3* wr = w + ((co * 3 + kr) * 3 + ks) * Ci;
      for (int ci = 0; ci < Ci; ++ci) acc += float(xs[ci]) * float(wr[ci]);
    }
  float v = acc * alpha + __bfloat162float(bias[co]);
  y[idx] = __float2bfloat16(v);
}

// ---- FP8 2D 3x3 SAME conv, output NCDHW [B, Co, T, H, W] ----
__global__ void fp8_conv2d_ncdhw_kernel(
    const __nv_fp8_e4m3* __restrict__ x,
    const __nv_fp8_e4m3* __restrict__ w,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ y,
    int B, int T, int H, int W, int Ci, int Co, float alpha) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B * Co * T * H * W;
  if (idx >= total) return;
  int co = idx % Co;
  int t = (idx / Co) % T;
  int h = (idx / (Co * T)) % H;
  int w_out = (idx / (Co * T * H)) % W;
  int n = idx / (Co * T * H * W);
  float acc = 0.0f;
  for (int kr = 0; kr < 3; ++kr)
    for (int ks = 0; ks < 3; ++ks) {
      int hi = h + kr - 1, wi = w_out + ks - 1;
      if (hi < 0 || hi >= H || wi < 0 || wi >= W) continue;
      const __nv_fp8_e4m3* xs = x + ((((n * T + t) * H + hi) * W + wi)) * Ci;
      const __nv_fp8_e4m3* wr = w + ((co * 3 + kr) * 3 + ks) * Ci;
      for (int ci = 0; ci < Ci; ++ci) acc += float(xs[ci]) * float(wr[ci]);
    }
  float v = acc * alpha + __bfloat162float(bias[co]);
  y[idx] = __float2bfloat16(v);
}

// ---- NVFP4 causal conv3d. out NDHWC [N, T_new, H, W, Co] (+ optional outer) ----
__global__ void fp4_conv3d_ndhwc_kernel(
    const uint8_t* __restrict__ cache_x, const uint8_t* __restrict__ new_x,
    const uint8_t* __restrict__ w,
    const uint8_t* __restrict__ cache_sf, const uint8_t* __restrict__ new_sf,
    const uint8_t* __restrict__ w_sf,
    const float* __restrict__ outer_w, const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ y,
    int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
    float alpha) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * T_new * H * W * Co;
  if (idx >= total) return;
  int co = idx % Co;
  int w_out = (idx / Co) % W;
  int h = (idx / (Co * W)) % H;
  int t = (idx / (Co * W * H)) % T_new;
  int b = idx / (Co * W * H * T_new);
  int CiHalf = Ci / 2;
  int CiBlk = Ci / 16;
  float acc = 0.0f;
  for (int kt = 0; kt < 3; ++kt)
    for (int kr = 0; kr < 3; ++kr)
      for (int ks = 0; ks < 3; ++ks) {
        int d = t + kt;
        int hi = h + kr - 1, wi = w_out + ks - 1;
        if (hi < 0 || hi >= H || wi < 0 || wi >= W) continue;
        if (d >= T_cache + T_new) continue;
        const uint8_t* xs;
        const uint8_t* xs_sf;
        if (d < T_cache) {
          int base = ((b * T_cache + d) * H + hi) * W + wi;
          xs = cache_x + base * CiHalf;
          xs_sf = cache_sf + base * CiBlk;
        } else {
          int dn = d - T_cache;
          int base = ((b * T_new + dn) * H + hi) * W + wi;
          xs = new_x + base * CiHalf;
          xs_sf = new_sf + base * CiBlk;
        }
        const uint8_t* wrow = w + ((((co * 3 + kt) * 3 + kr) * 3 + ks)) * CiHalf;
        const uint8_t* wrow_sf = w_sf + ((((co * 3 + kt) * 3 + kr) * 3 + ks)) * CiBlk;
        for (int ci = 0; ci < Ci; ++ci) {
          float a = fp4_read(xs, ci) * ue4_read(xs_sf, ci >> 4);
          float bb = fp4_read(wrow, ci) * ue4_read(wrow_sf, ci >> 4);
          acc += a * bb;
        }
      }
  float scale = alpha;
  if (outer_w != nullptr) scale = outer_w[co] * alpha;
  float v = acc * scale + __bfloat162float(bias[co]);
  y[idx] = __float2bfloat16(v);
}

// ---- NVFP4 causal conv3d + residual. out NCDHW [N, Co, T_new, H, W] ----
__global__ void fp4_conv3d_ncdhw_res_kernel(
    const uint8_t* __restrict__ cache_x, const uint8_t* __restrict__ new_x,
    const uint8_t* __restrict__ w,
    const uint8_t* __restrict__ cache_sf, const uint8_t* __restrict__ new_sf,
    const uint8_t* __restrict__ w_sf,
    const float* __restrict__ outer_w, const __nv_bfloat16* __restrict__ bias,
    const __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ y,
    int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
    float alpha) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * Co * T_new * H * W;
  if (idx >= total) return;
  int co = idx % Co;
  int t = (idx / Co) % T_new;
  int h = (idx / (Co * T_new)) % H;
  int w_out = (idx / (Co * T_new * H)) % W;
  int b = idx / (Co * T_new * H * W);
  int CiHalf = Ci / 2;
  int CiBlk = Ci / 16;
  float acc = 0.0f;
  for (int kt = 0; kt < 3; ++kt)
    for (int kr = 0; kr < 3; ++kr)
      for (int ks = 0; ks < 3; ++ks) {
        int d = t + kt;
        int hi = h + kr - 1, wi = w_out + ks - 1;
        if (hi < 0 || hi >= H || wi < 0 || wi >= W) continue;
        if (d >= T_cache + T_new) continue;
        const uint8_t* xs;
        const uint8_t* xs_sf;
        if (d < T_cache) {
          int base = ((b * T_cache + d) * H + hi) * W + wi;
          xs = cache_x + base * CiHalf;
          xs_sf = cache_sf + base * CiBlk;
        } else {
          int dn = d - T_cache;
          int base = ((b * T_new + dn) * H + hi) * W + wi;
          xs = new_x + base * CiHalf;
          xs_sf = new_sf + base * CiBlk;
        }
        const uint8_t* wrow = w + ((((co * 3 + kt) * 3 + kr) * 3 + ks)) * CiHalf;
        const uint8_t* wrow_sf = w_sf + ((((co * 3 + kt) * 3 + kr) * 3 + ks)) * CiBlk;
        for (int ci = 0; ci < Ci; ++ci) {
          float a = fp4_read(xs, ci) * ue4_read(xs_sf, ci >> 4);
          float bb = fp4_read(wrow, ci) * ue4_read(wrow_sf, ci >> 4);
          acc += a * bb;
        }
      }
  float scale = alpha;
  if (outer_w != nullptr) scale = outer_w[co] * alpha;
  float v = acc * scale + __bfloat162float(bias[co]);
  v += __bfloat162float(residual[idx]);
  y[idx] = __float2bfloat16(v);
}

template <int N_ARGS_PAD>
void launch_fp8_conv3d_ndhwc(
    const void* cache_x, const void* new_x, const void* w, void* y,
    const void* bias, int N, int tc, int tn, int H, int W, int Ci, int Co,
    float alpha, cudaStream_t stream) {
  (void)N_ARGS_PAD;
  int total = N * tn * H * W * Co;
  fp8_conv3d_ndhwc_kernel<<<(total + THREADS - 1) / THREADS, THREADS, 0,
                             stream>>>(
      reinterpret_cast<const __nv_fp8_e4m3*>(cache_x),
      reinterpret_cast<const __nv_fp8_e4m3*>(new_x),
      reinterpret_cast<const __nv_fp8_e4m3*>(w),
      reinterpret_cast<const __nv_bfloat16*>(bias),
      reinterpret_cast<__nv_bfloat16*>(y),
      N, tc, tn, H, W, Ci, Co, alpha);
}

}  // namespace

int fp8_conv3d_v18_ncdhw_res_bf16out_simt(
    const void* cache_x, const void* new_x, const void* w, void* y,
    const void* bias, const void* residual,
    int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
    float alpha, cudaStream_t stream) {
  if (N <= 0 || T_new <= 0 || H <= 0 || W <= 0 || Ci <= 0 || Co <= 0) return -1;
  int total = N * Co * T_new * H * W;
  fp8_conv3d_ncdhw_res_kernel<<<(total + THREADS - 1) / THREADS, THREADS, 0,
                                stream>>>(
      reinterpret_cast<const __nv_fp8_e4m3*>(cache_x),
      reinterpret_cast<const __nv_fp8_e4m3*>(new_x),
      reinterpret_cast<const __nv_fp8_e4m3*>(w),
      reinterpret_cast<const __nv_bfloat16*>(bias),
      reinterpret_cast<const __nv_bfloat16*>(residual),
      reinterpret_cast<__nv_bfloat16*>(y),
      N, T_cache, T_new, H, W, Ci, Co, alpha);
  return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int fp8_conv3d_v17_ndhwc_bf16out_simt(
    const void* cache_x, const void* new_x, const void* w, void* y,
    const void* bias,
    int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
    float alpha, cudaStream_t stream) {
  if (N <= 0 || T_new <= 0 || H <= 0 || W <= 0 || Ci <= 0 || Co <= 0) return -1;
  launch_fp8_conv3d_ndhwc<0>(cache_x, new_x, w, y, bias, N, T_cache, T_new, H,
                             W, Ci, Co, alpha, stream);
  return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int fp8_conv3d_v17_anyco_ndhwc_bf16out_simt(
    const void* cache_x, const void* new_x, const void* w, void* y,
    const void* bias,
    int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
    float alpha, cudaStream_t stream) {
  return fp8_conv3d_v17_ndhwc_bf16out_simt(
      cache_x, new_x, w, y, bias, N, T_cache, T_new, H, W, Ci, Co, alpha,
      stream);
}

int fp8_conv2d_3x3_v2_nhwc_bf16out_simt(
    const void* x, const void* w, void* y, const void* bias,
    int N, int H, int W, int Ci, int Co, float alpha, cudaStream_t stream) {
  if (N <= 0 || H <= 0 || W <= 0 || Ci <= 0 || Co <= 0) return -1;
  int total = N * H * W * Co;
  fp8_conv2d_nhwc_kernel<<<(total + THREADS - 1) / THREADS, THREADS, 0,
                           stream>>>(
      reinterpret_cast<const __nv_fp8_e4m3*>(x),
      reinterpret_cast<const __nv_fp8_e4m3*>(w),
      reinterpret_cast<const __nv_bfloat16*>(bias),
      reinterpret_cast<__nv_bfloat16*>(y),
      N, H, W, Ci, Co, alpha);
  return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int fp8_conv2d_3x3_v2_nhwc_ncdhw_bf16out_simt(
    const void* x, const void* w, void* y, const void* bias,
    int B, int T, int H, int W, int Ci, int Co, float alpha,
    cudaStream_t stream) {
  if (B <= 0 || T <= 0 || H <= 0 || W <= 0 || Ci <= 0 || Co <= 0) return -1;
  int total = B * Co * T * H * W;
  fp8_conv2d_ncdhw_kernel<<<(total + THREADS - 1) / THREADS, THREADS, 0,
                            stream>>>(
      reinterpret_cast<const __nv_fp8_e4m3*>(x),
      reinterpret_cast<const __nv_fp8_e4m3*>(w),
      reinterpret_cast<const __nv_bfloat16*>(bias),
      reinterpret_cast<__nv_bfloat16*>(y),
      B, T, H, W, Ci, Co, alpha);
  return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int fp4_conv3d_ndhwc_bf16out_simt(
    const void* cache_x, const void* new_x, const void* w,
    const void* cache_sf, const void* new_sf, const void* w_sf,
    const void* outer_w, void* y, const void* bias,
    int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
    float alpha, cudaStream_t stream) {
  if (N <= 0 || T_new <= 0 || H <= 0 || W <= 0 || Ci <= 0 || Co <= 0) return -1;
  int total = N * T_new * H * W * Co;
  fp4_conv3d_ndhwc_kernel<<<(total + THREADS - 1) / THREADS, THREADS, 0,
                            stream>>>(
      reinterpret_cast<const uint8_t*>(cache_x),
      reinterpret_cast<const uint8_t*>(new_x),
      reinterpret_cast<const uint8_t*>(w),
      reinterpret_cast<const uint8_t*>(cache_sf),
      reinterpret_cast<const uint8_t*>(new_sf),
      reinterpret_cast<const uint8_t*>(w_sf),
      reinterpret_cast<const float*>(outer_w),
      reinterpret_cast<const __nv_bfloat16*>(bias),
      reinterpret_cast<__nv_bfloat16*>(y),
      N, T_cache, T_new, H, W, Ci, Co, alpha);
  return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int fp4_conv3d_ncdhw_res_bf16out_simt(
    const void* cache_x, const void* new_x, const void* w,
    const void* cache_sf, const void* new_sf, const void* w_sf,
    const void* outer_w, void* y, const void* bias, const void* residual,
    int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
    float alpha, cudaStream_t stream) {
  if (N <= 0 || T_new <= 0 || H <= 0 || W <= 0 || Ci <= 0 || Co <= 0) return -1;
  int total = N * Co * T_new * H * W;
  fp4_conv3d_ncdhw_res_kernel<<<(total + THREADS - 1) / THREADS, THREADS, 0,
                                stream>>>(
      reinterpret_cast<const uint8_t*>(cache_x),
      reinterpret_cast<const uint8_t*>(new_x),
      reinterpret_cast<const uint8_t*>(w),
      reinterpret_cast<const uint8_t*>(cache_sf),
      reinterpret_cast<const uint8_t*>(new_sf),
      reinterpret_cast<const uint8_t*>(w_sf),
      reinterpret_cast<const float*>(outer_w),
      reinterpret_cast<const __nv_bfloat16*>(bias),
      reinterpret_cast<const __nv_bfloat16*>(residual),
      reinterpret_cast<__nv_bfloat16*>(y),
      N, T_cache, T_new, H, W, Ci, Co, alpha);
  return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

}  // namespace conv
}  // namespace flash_rt
