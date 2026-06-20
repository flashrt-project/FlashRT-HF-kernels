// SPDX-License-Identifier: Apache-2.0

#include "diffusion_step_ops.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace diffusion_step_ops {
namespace {

template <typename T>
__device__ __forceinline__ float to_float(T value);

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <>
__device__ __forceinline__ float to_float<__half>(__half value) {
  return __half2float(value);
}

template <typename T>
__device__ __forceinline__ T from_float(float value);

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float value) {
  return __float2bfloat16(value);
}

template <>
__device__ __forceinline__ __half from_float<__half>(float value) {
  return __float2half(value);
}

int launch_blocks(int64_t n) {
  int64_t blocks = (n + 255) / 256;
  if (blocks > 4096) blocks = 4096;
  return static_cast<int>(blocks);
}

template <typename T>
__global__ void add_kernel(const T* __restrict__ a,
                           const T* __restrict__ b,
                           T* __restrict__ out,
                           int64_t n) {
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < n;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    out[idx] = from_float<T>(to_float<T>(a[idx]) + to_float<T>(b[idx]));
  }
}

__global__ void euler_step_bf16_kernel(const __nv_bfloat16* __restrict__ latent,
                                       const __nv_bfloat16* __restrict__ velocity,
                                       __nv_bfloat16* __restrict__ out,
                                       float dt,
                                       int64_t n) {
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < n;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float value = __bfloat162float(latent[idx]) + __bfloat162float(velocity[idx]) * dt;
    out[idx] = __float2bfloat16(value);
  }
}

template <typename T>
__global__ void cfg_combine_kernel(T* __restrict__ residual,
                                   const T* __restrict__ v_cond,
                                   const T* __restrict__ v_uncond,
                                   float beta,
                                   int64_t n) {
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < n;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float r = to_float<T>(residual[idx]);
    const float vc = to_float<T>(v_cond[idx]);
    const float vu = to_float<T>(v_uncond[idx]);
    residual[idx] = from_float<T>(r + vu + beta * (vc - vu));
  }
}

__global__ void teacher_force_first_frame_bf16_kernel(
    __nv_bfloat16* __restrict__ video_latent,
    const __nv_bfloat16* __restrict__ cond_latent,
    int c,
    int t,
    int h,
    int w,
    int64_t n) {
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < n;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int64_t q = idx;
    const int ww = static_cast<int>(q % w);
    q /= w;
    const int hh = static_cast<int>(q % h);
    q /= h;
    const int cc = static_cast<int>(q % c);
    q /= c;
    const int64_t bb = q;
    const int64_t dst = (((bb * c + cc) * static_cast<int64_t>(t)) * h + hh) * w + ww;
    video_latent[dst] = cond_latent[idx];
  }
}

__global__ void motus_decode_postprocess_bf16_to_fp32_kernel(
    const __nv_bfloat16* __restrict__ decoded,
    float* __restrict__ out,
    int c,
    int t_in,
    int h,
    int w,
    int64_t n) {
  const int t_out = t_in - 1;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < n;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int64_t q = idx;
    const int ww = static_cast<int>(q % w);
    q /= w;
    const int hh = static_cast<int>(q % h);
    q /= h;
    const int tt = static_cast<int>(q % t_out);
    q /= t_out;
    const int cc = static_cast<int>(q % c);
    q /= c;
    const int64_t bb = q;
    const int64_t src = (((bb * c + cc) * static_cast<int64_t>(t_in) + (tt + 1)) * h + hh) * w + ww;
    float value = (__bfloat162float(decoded[src]) + 1.0f) * 0.5f;
    value = fminf(fmaxf(value, 0.0f), 1.0f);
    out[idx] = value;
  }
}

__global__ void cast_bf16_to_fp32_kernel(const __nv_bfloat16* __restrict__ src,
                                         float* __restrict__ dst,
                                         int64_t n) {
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < n;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    dst[idx] = __bfloat162float(src[idx]);
  }
}

}  // namespace

void add_bf16_out(const void* a, const void* b, void* out, int64_t n, cudaStream_t stream) {
  add_kernel<__nv_bfloat16><<<launch_blocks(n), 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(a),
      reinterpret_cast<const __nv_bfloat16*>(b),
      reinterpret_cast<__nv_bfloat16*>(out),
      n);
}

void euler_step_bf16_out(
    const void* latent,
    const void* velocity,
    void* out,
    float dt,
    int64_t n,
    cudaStream_t stream) {
  euler_step_bf16_kernel<<<launch_blocks(n), 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(latent),
      reinterpret_cast<const __nv_bfloat16*>(velocity),
      reinterpret_cast<__nv_bfloat16*>(out),
      dt,
      n);
}

void cfg_combine_into_residual_bf16(
    void* residual,
    const void* v_cond,
    const void* v_uncond,
    float beta,
    int64_t n,
    cudaStream_t stream) {
  cfg_combine_kernel<__nv_bfloat16><<<launch_blocks(n), 256, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(residual),
      reinterpret_cast<const __nv_bfloat16*>(v_cond),
      reinterpret_cast<const __nv_bfloat16*>(v_uncond),
      beta,
      n);
}

void cfg_combine_into_residual_fp16(
    void* residual,
    const void* v_cond,
    const void* v_uncond,
    float beta,
    int64_t n,
    cudaStream_t stream) {
  cfg_combine_kernel<__half><<<launch_blocks(n), 256, 0, stream>>>(
      reinterpret_cast<__half*>(residual),
      reinterpret_cast<const __half*>(v_cond),
      reinterpret_cast<const __half*>(v_uncond),
      beta,
      n);
}

void teacher_force_first_frame_bf16(
    void* video_latent,
    const void* cond_latent,
    int b,
    int c,
    int t,
    int h,
    int w,
    cudaStream_t stream) {
  const int64_t n = static_cast<int64_t>(b) * c * h * w;
  teacher_force_first_frame_bf16_kernel<<<launch_blocks(n), 256, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(video_latent),
      reinterpret_cast<const __nv_bfloat16*>(cond_latent),
      c,
      t,
      h,
      w,
      n);
}

void motus_decode_postprocess_bf16_to_fp32(
    const void* decoded,
    void* out,
    int b,
    int c,
    int t_in,
    int h,
    int w,
    cudaStream_t stream) {
  const int64_t n = static_cast<int64_t>(b) * c * (t_in - 1) * h * w;
  motus_decode_postprocess_bf16_to_fp32_kernel<<<launch_blocks(n), 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(decoded),
      reinterpret_cast<float*>(out),
      c,
      t_in,
      h,
      w,
      n);
}

void cast_bf16_to_fp32(const void* src, void* dst, int64_t n, cudaStream_t stream) {
  cast_bf16_to_fp32_kernel<<<launch_blocks(n), 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(src),
      reinterpret_cast<float*>(dst),
      n);
}

}  // namespace diffusion_step_ops
}  // namespace flash_rt
