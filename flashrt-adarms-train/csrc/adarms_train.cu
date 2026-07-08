// SPDX-License-Identifier: Apache-2.0
//
// AdaRMS + gated-residual training kernels (forward and backward).
//
// Operator (fp32 internal math, matching the PyTorch reference):
//   rstd = rsqrt(mean(x^2) + eps)
//   xhat = x * rstd
//   adaptive:      y = xhat * (1 + scale) + shift
//   non-adaptive:  y = xhat * (1 + weight)
// resgate variant computes r = x + h * gate (gate optional) first, keeps r
// as a real output, and normalizes r.
//
// One CTA per row; the row is cached in registers (vectorized 16-byte
// loads), reduced with warp shuffles, and written back in one pass.

#include <cuda_bf16.h>

#include "adarms_train.cuh"

namespace flashrt_hub {
namespace adarms_train {

namespace {

constexpr int kThreads = 256;
constexpr int kMaxVecsPerThread = 4;  // supports H up to 8192 (bf16) / 4096 (f32)

template <typename T>
struct VecTraits;

template <>
struct VecTraits<__nv_bfloat16> {
  static constexpr int kLanes = 8;  // 16 bytes
};

template <>
struct VecTraits<float> {
  static constexpr int kLanes = 4;  // 16 bytes
};

template <typename T>
__device__ __forceinline__ float to_f32(T v);
template <>
__device__ __forceinline__ float to_f32(__nv_bfloat16 v) {
  return __bfloat162float(v);
}
template <>
__device__ __forceinline__ float to_f32(float v) {
  return v;
}

template <typename T>
__device__ __forceinline__ T from_f32(float v);
template <>
__device__ __forceinline__ __nv_bfloat16 from_f32(float v) {
  return __float2bfloat16(v);
}
template <>
__device__ __forceinline__ float from_f32(float v) {
  return v;
}

__device__ __forceinline__ float block_reduce_sum(float v) {
  __shared__ float warp_sums[kThreads / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, off);
  }
  if (lane == 0) warp_sums[warp] = v;
  __syncthreads();
  v = (threadIdx.x < kThreads / 32) ? warp_sums[threadIdx.x] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int off = kThreads / 64; off > 0; off >>= 1) {
      v += __shfl_down_sync(0xffffffffu, v, off);
    }
    if (lane == 0) warp_sums[0] = v;
  }
  __syncthreads();
  return warp_sums[0];
}

template <typename T>
__device__ __forceinline__ void load_vec(const T* row, int idx, float* out) {
  constexpr int kLanes = VecTraits<T>::kLanes;
  uint4 raw = *reinterpret_cast<const uint4*>(row + idx);
  const T* vals = reinterpret_cast<const T*>(&raw);
#pragma unroll
  for (int i = 0; i < kLanes; ++i) out[i] = to_f32(vals[i]);
}

template <typename T>
__device__ __forceinline__ void store_vec(T* row, int idx, const float* in) {
  constexpr int kLanes = VecTraits<T>::kLanes;
  uint4 raw;
  T* vals = reinterpret_cast<T*>(&raw);
#pragma unroll
  for (int i = 0; i < kLanes; ++i) vals[i] = from_f32<T>(in[i]);
  *reinterpret_cast<uint4*>(row + idx) = raw;
}

template <typename T>
__device__ __forceinline__ const T* mod_row(ModView m, int b, int t) {
  return reinterpret_cast<const T*>(m.ptr) + (long)b * m.batch_stride +
         (long)t * m.token_stride;
}

// ---------------------------------------------------------------------------
// forward
// ---------------------------------------------------------------------------

template <typename T, bool kAdaptive, bool kResGate, int kNVec>
__global__ void adarms_fwd_kernel(const T* __restrict__ x,
                                  const T* __restrict__ hbr,
                                  const T* __restrict__ gate, ModView scale,
                                  ModView shift, const T* __restrict__ weight,
                                  T* __restrict__ r, T* __restrict__ y,
                                  float* __restrict__ rstd_out, int tokens,
                                  int h, float eps) {
  constexpr int kLanes = VecTraits<T>::kLanes;
  const int row = blockIdx.x;
  const int b = row / tokens;
  const int t = row - b * tokens;
  const T* xr = x + (long)row * h;

  float vals[kNVec][kLanes];
  float sq = 0.0f;
  int nvec = 0;
  for (int idx = threadIdx.x * kLanes; idx < h; idx += kThreads * kLanes, ++nvec) {
    load_vec<T>(xr, idx, vals[nvec]);
    if (kResGate) {
      // Match the reference rounding sequence exactly: h*gate rounds to the
      // io dtype first, then the add rounds again, so r is bitwise-identical
      // to `x + h * gate` computed on io-dtype tensors.
      float hv[kLanes];
      load_vec<T>(hbr + (long)row * h, idx, hv);
      if (gate != nullptr) {
        float gv[kLanes];
        load_vec<T>(gate + (long)row * h, idx, gv);
#pragma unroll
        for (int i = 0; i < kLanes; ++i) {
          const float prod = to_f32(from_f32<T>(hv[i] * gv[i]));
          vals[nvec][i] = to_f32(from_f32<T>(vals[nvec][i] + prod));
        }
      } else {
#pragma unroll
        for (int i = 0; i < kLanes; ++i)
          vals[nvec][i] = to_f32(from_f32<T>(vals[nvec][i] + hv[i]));
      }
      store_vec<T>(r + (long)row * h, idx, vals[nvec]);
    }
#pragma unroll
    for (int i = 0; i < kLanes; ++i) sq += vals[nvec][i] * vals[nvec][i];
  }

  const float rstd = rsqrtf(block_reduce_sum(sq) / (float)h + eps);
  if (threadIdx.x == 0) rstd_out[row] = rstd;

  const T* srow = kAdaptive ? mod_row<T>(scale, b, t) : nullptr;
  const T* frow = kAdaptive ? mod_row<T>(shift, b, t) : nullptr;
  int v = 0;
  for (int idx = threadIdx.x * kLanes; idx < h; idx += kThreads * kLanes, ++v) {
    float sc[kLanes], sf[kLanes];
    if (kAdaptive) {
      load_vec<T>(srow, idx, sc);
      load_vec<T>(frow, idx, sf);
    } else {
      load_vec<T>(weight, idx, sc);
    }
    float out[kLanes];
#pragma unroll
    for (int i = 0; i < kLanes; ++i) {
      const float xhat = vals[v][i] * rstd;
      out[i] = kAdaptive ? xhat * (1.0f + sc[i]) + sf[i] : xhat * (1.0f + sc[i]);
    }
    store_vec<T>(y + (long)row * h, idx, out);
  }
}

// ---------------------------------------------------------------------------
// backward
// ---------------------------------------------------------------------------
//   dyh = dy * (1 + scale_or_weight)
//   c   = mean(dyh * xhat)
//   dnorm = rstd * (dyh - xhat * c)
//   site1:   dx = dnorm
//   resgate: dr_total = dnorm + dyr; dh = dr_total * gate; dg = dr_total * h
//   adaptive: dscale_elem = dy * xhat (per row; caller reduces broadcasts)
//   non-adaptive: dweight += sum_rows(dy * xhat) via fp32 atomics

// Adaptive mode: one CTA per row, dscale written per row. Non-adaptive
// (weight) mode: grid-stride CTAs accumulate the weight grad in registers
// across their rows and write ONE partial row per CTA — the (grid, H)
// partials are summed by the caller (per-column atomics across thousands of
// rows serialize badly).
template <typename T, bool kAdaptive, bool kResGate, int kNVec>
__global__ void adarms_bwd_kernel(
    const T* __restrict__ dy, const T* __restrict__ dyr,
    const T* __restrict__ xin, const T* __restrict__ hbr,
    const T* __restrict__ gate, ModView scale, const T* __restrict__ weight,
    const float* __restrict__ rstd_in, T* __restrict__ dx,
    T* __restrict__ dh, T* __restrict__ dg, T* __restrict__ dscale_elem,
    int rows, int tokens, int h) {
  constexpr int kLanes = VecTraits<T>::kLanes;

  // Weight mode accumulates the weight grad in registers across this CTA's
  // grid-stride rows and writes one fp32 partial row at the end (summed by
  // the caller); materializing per-row elements would cost an extra full
  // read+write of the activation. Adaptive mode launches one CTA per row.
  float wv[kNVec][kLanes];
  float dw[kNVec][kLanes];
  if (!kAdaptive) {
    int nv = 0;
    for (int idx = threadIdx.x * kLanes; idx < h; idx += kThreads * kLanes, ++nv) {
      load_vec<T>(weight, idx, wv[nv]);
#pragma unroll
      for (int i = 0; i < kLanes; ++i) dw[nv][i] = 0.0f;
    }
  }

  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const int b = row / tokens;
    const int t = row - b * tokens;
    const float rstd = rstd_in[row];

    float xv[kNVec][kLanes];
    float dyv[kNVec][kLanes];
    float scv[kNVec][kLanes];
    const T* srow = kAdaptive ? mod_row<T>(scale, b, t) : nullptr;

    float dot = 0.0f;
    int nvec = 0;
    for (int idx = threadIdx.x * kLanes; idx < h; idx += kThreads * kLanes, ++nvec) {
      load_vec<T>(xin + (long)row * h, idx, xv[nvec]);
      load_vec<T>(dy + (long)row * h, idx, dyv[nvec]);
      if (kAdaptive) load_vec<T>(srow, idx, scv[nvec]);
#pragma unroll
      for (int i = 0; i < kLanes; ++i) {
        const float sc = kAdaptive ? scv[nvec][i] : wv[nvec][i];
        const float xhat = xv[nvec][i] * rstd;
        dot += dyv[nvec][i] * (1.0f + sc) * xhat;
      }
    }
    const float c = block_reduce_sum(dot) / (float)h;

    int v = 0;
    for (int idx = threadIdx.x * kLanes; idx < h; idx += kThreads * kLanes, ++v) {
      float out[kLanes];
      float dsc[kLanes];
#pragma unroll
      for (int i = 0; i < kLanes; ++i) {
        const float sc = kAdaptive ? scv[v][i] : wv[v][i];
        const float xhat = xv[v][i] * rstd;
        const float dyh = dyv[v][i] * (1.0f + sc);
        out[i] = rstd * (dyh - xhat * c);
        dsc[i] = dyv[v][i] * xhat;
      }
      if (kResGate) {
        if (dyr != nullptr) {
          float rv[kLanes];
          load_vec<T>(dyr + (long)row * h, idx, rv);
#pragma unroll
          for (int i = 0; i < kLanes; ++i) out[i] += rv[i];
        }
        store_vec<T>(dx + (long)row * h, idx, out);  // dr_total (== dx)
        float hv[kLanes];
        load_vec<T>(hbr + (long)row * h, idx, hv);
        if (gate != nullptr) {
          float gv[kLanes], dhv[kLanes], dgv[kLanes];
          load_vec<T>(gate + (long)row * h, idx, gv);
#pragma unroll
          for (int i = 0; i < kLanes; ++i) {
            dhv[i] = out[i] * gv[i];
            dgv[i] = out[i] * hv[i];
          }
          store_vec<T>(dh + (long)row * h, idx, dhv);
          store_vec<T>(dg + (long)row * h, idx, dgv);
        } else {
          store_vec<T>(dh + (long)row * h, idx, out);
        }
      } else {
        store_vec<T>(dx + (long)row * h, idx, out);
      }
      if (kAdaptive) {
        store_vec<T>(dscale_elem + (long)row * h, idx, dsc);
      } else {
#pragma unroll
        for (int i = 0; i < kLanes; ++i) dw[v][i] += dsc[i];
      }
    }
    if (!kAdaptive) __syncthreads();  // block_reduce smem reuse across rows
  }

  if (!kAdaptive) {
    float* prow = reinterpret_cast<float*>(dscale_elem) + (long)blockIdx.x * h;
    int v = 0;
    for (int idx = threadIdx.x * kLanes; idx < h; idx += kThreads * kLanes, ++v) {
#pragma unroll
      for (int i = 0; i < kLanes; ++i) prow[idx + i] = dw[v][i];
    }
  }
}

template <typename T, bool A, bool R, typename... Args>
void launch_fwd_nvec(int nvec, dim3 grid, dim3 block, cudaStream_t stream,
                     Args... args) {
  switch (nvec) {
    case 1: adarms_fwd_kernel<T, A, R, 1><<<grid, block, 0, stream>>>(args...); break;
    case 2: adarms_fwd_kernel<T, A, R, 2><<<grid, block, 0, stream>>>(args...); break;
    case 3: adarms_fwd_kernel<T, A, R, 3><<<grid, block, 0, stream>>>(args...); break;
    default: adarms_fwd_kernel<T, A, R, 4><<<grid, block, 0, stream>>>(args...); break;
  }
}

template <typename T, bool A, bool R, typename... Args>
void launch_bwd_nvec(int nvec, dim3 grid, dim3 block, cudaStream_t stream,
                     Args... args) {
  switch (nvec) {
    case 1: adarms_bwd_kernel<T, A, R, 1><<<grid, block, 0, stream>>>(args...); break;
    case 2: adarms_bwd_kernel<T, A, R, 2><<<grid, block, 0, stream>>>(args...); break;
    case 3: adarms_bwd_kernel<T, A, R, 3><<<grid, block, 0, stream>>>(args...); break;
    default: adarms_bwd_kernel<T, A, R, 4><<<grid, block, 0, stream>>>(args...); break;
  }
}

template <typename T>
void fwd_dispatch(const void* x, const void* hbr, const void* gate,
                  ModView scale, ModView shift, const void* weight, void* r,
                  void* y, float* rstd, int rows, int tokens, int h, float eps,
                  bool resgate, cudaStream_t stream) {
  const bool adaptive = scale.ptr != nullptr;
  const int nvec = (h + kThreads * VecTraits<T>::kLanes - 1) /
                   (kThreads * VecTraits<T>::kLanes);
  dim3 grid(rows), block(kThreads);
#define LAUNCH(A, R)                                                          \
  launch_fwd_nvec<T, A, R>(nvec, grid, block, stream,                     \
      reinterpret_cast<const T*>(x), reinterpret_cast<const T*>(hbr),         \
      reinterpret_cast<const T*>(gate), scale, shift,                         \
      reinterpret_cast<const T*>(weight), reinterpret_cast<T*>(r),            \
      reinterpret_cast<T*>(y), rstd, tokens, h, eps)
  if (adaptive && resgate) LAUNCH(true, true);
  else if (adaptive) LAUNCH(true, false);
  else if (resgate) LAUNCH(false, true);
  else LAUNCH(false, false);
#undef LAUNCH
}

template <typename T>
void bwd_dispatch(const void* dy, const void* dyr, const void* xin,
                  const void* hbr, const void* gate, ModView scale,
                  const void* weight, const float* rstd, void* dx, void* dh,
                  void* dg, void* dscale_elem, int rows,
                  int tokens, int h, bool resgate, cudaStream_t stream) {
  const bool adaptive = scale.ptr != nullptr;
  const int nvec = (h + kThreads * VecTraits<T>::kLanes - 1) /
                   (kThreads * VecTraits<T>::kLanes);
  dim3 grid(adaptive ? rows : bwd_weight_grid(rows)), block(kThreads);
#define LAUNCH(A, R)                                                          \
  launch_bwd_nvec<T, A, R>(nvec, grid, block, stream,                     \
      reinterpret_cast<const T*>(dy), reinterpret_cast<const T*>(dyr),        \
      reinterpret_cast<const T*>(xin), reinterpret_cast<const T*>(hbr),       \
      reinterpret_cast<const T*>(gate), scale,                                \
      reinterpret_cast<const T*>(weight), rstd, reinterpret_cast<T*>(dx),     \
      reinterpret_cast<T*>(dh), reinterpret_cast<T*>(dg),                     \
      reinterpret_cast<T*>(dscale_elem), rows, tokens, h)
  if (adaptive && resgate) LAUNCH(true, true);
  else if (adaptive) LAUNCH(true, false);
  else if (resgate) LAUNCH(false, true);
  else LAUNCH(false, false);
#undef LAUNCH
}

}  // namespace

void adarms_fwd_launch(const void* x, ModView scale, ModView shift,
                       const void* weight, void* y, float* rstd, int rows,
                       int tokens, int h, float eps, bool bf16,
                       cudaStream_t stream) {
  if (bf16)
    fwd_dispatch<__nv_bfloat16>(x, nullptr, nullptr, scale, shift, weight,
                                nullptr, y, rstd, rows, tokens, h, eps, false,
                                stream);
  else
    fwd_dispatch<float>(x, nullptr, nullptr, scale, shift, weight, nullptr, y,
                        rstd, rows, tokens, h, eps, false, stream);
}

void adarms_bwd_launch(const void* dy, const void* x, ModView scale,
                       const void* weight, const float* rstd, void* dx,
                       void* dscale_elem, int rows, int tokens,
                       int h, bool bf16, cudaStream_t stream) {
  if (bf16)
    bwd_dispatch<__nv_bfloat16>(dy, nullptr, x, nullptr, nullptr, scale,
                                weight, rstd, dx, nullptr, nullptr,
                                dscale_elem, rows, tokens, h, false, stream);
  else
    bwd_dispatch<float>(dy, nullptr, x, nullptr, nullptr, scale, weight, rstd,
                        dx, nullptr, nullptr, dscale_elem, rows,
                        tokens, h, false, stream);
}

void resgate_adarms_fwd_launch(const void* x, const void* hbr,
                               const void* gate, ModView scale, ModView shift,
                               const void* weight, void* r, void* y,
                               float* rstd, int rows, int tokens, int h,
                               float eps, bool bf16, cudaStream_t stream) {
  if (bf16)
    fwd_dispatch<__nv_bfloat16>(x, hbr, gate, scale, shift, weight, r, y, rstd,
                                rows, tokens, h, eps, true, stream);
  else
    fwd_dispatch<float>(x, hbr, gate, scale, shift, weight, r, y, rstd, rows,
                        tokens, h, eps, true, stream);
}

void resgate_adarms_bwd_launch(const void* dy, const void* dyr, const void* r,
                               const void* hbr, const void* gate,
                               ModView scale, const void* weight,
                               const float* rstd, void* dr_total, void* dh,
                               void* dg, void* dscale_elem,
                               int rows, int tokens, int h, bool bf16,
                               cudaStream_t stream) {
  if (bf16)
    bwd_dispatch<__nv_bfloat16>(dy, dyr, r, hbr, gate, scale, weight, rstd,
                                dr_total, dh, dg, dscale_elem, rows,
                                tokens, h, true, stream);
  else
    bwd_dispatch<float>(dy, dyr, r, hbr, gate, scale, weight, rstd, dr_total,
                        dh, dg, dscale_elem, rows, tokens, h, true,
                        stream);
}

}  // namespace adarms_train
}  // namespace flashrt_hub
