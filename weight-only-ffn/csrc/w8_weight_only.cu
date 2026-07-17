#include "weight_only_ffn.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace flashrt_weight_only {
namespace {

constexpr int kBM = 64;
constexpr int kBN = 64;
constexpr int kBK = 16;
constexpr int kKT = 64;
constexpr int kAS = kKT + 8;
constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;

__device__ __forceinline__ void cp16(void* dst, const void* src) {
  const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(address), "l"(src));
}

__device__ __forceinline__ uint32_t pack_bf16x2(float a, float b) {
  __nv_bfloat162 value = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&value);
}

__device__ __forceinline__ void mma(float& c0, float& c1, float& c2, float& c3,
                                    uint32_t a0, uint32_t a1,
                                    uint32_t a2, uint32_t a3,
                                    uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__global__ __launch_bounds__(kThreads) void w8a16_gemm_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int8_t* __restrict__ weight,
    const float* __restrict__ scales,
    __nv_bfloat16* __restrict__ out,
    int m, int n, int k) {
  const int bm = blockIdx.y * kBM;
  const int bn = blockIdx.x * kBN;
  __shared__ __nv_bfloat16 sa[2][kBM * kAS];
  __shared__ int8_t sw[2][kBN * kKT];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int warp_m = warp >> 1;
  const int warp_n = warp & 1;
  float accum[2][4][4] = {};
  const int tiles = k / kKT;

  auto load = [&](int buffer, int tile) {
    const int k0 = tile * kKT;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int index = (tid + j * kThreads) * 8;
      const int row = index / kKT;
      const int offset = index % kKT;
      const int global_row = min(bm + row, m - 1);
      cp16(&sa[buffer][row * kAS + offset],
           &x[static_cast<size_t>(global_row) * k + k0 + offset]);
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int index = (tid + j * kThreads) * 16;
      const int row = index / kKT;
      const int offset = index % kKT;
      const int global_row = min(bn + row, n - 1);
      cp16(&sw[buffer][index],
           &weight[static_cast<size_t>(global_row) * k + k0 + offset]);
    }
  };

  load(0, 0);
  asm volatile("cp.async.commit_group;\n" ::);
  if (tiles > 1) {
    load(1, 1);
    asm volatile("cp.async.commit_group;\n" ::);
  }

  const int fragment_row = lane >> 2;
  const int kk = (lane & 3) * 2;
  for (int tile = 0; tile < tiles; ++tile) {
    const int buffer = tile & 1;
    if (tile + 1 < tiles) asm volatile("cp.async.wait_group 1;\n" ::);
    else asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();

#pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
      const int kb = sub * kBK;
      uint32_t b0[4], b1[4];
#pragma unroll
      for (int jb = 0; jb < 4; ++jb) {
        const int local_n = warp_n * 32 + jb * 8 + fragment_row;
        const int global_n = min(bn + local_n, n - 1);
        const int8_t* row = &sw[buffer][local_n * kKT + kb + kk];
        const float scale = scales[global_n];
        b0[jb] = pack_bf16x2(float(row[0]) * scale, float(row[1]) * scale);
        b1[jb] = pack_bf16x2(float(row[8]) * scale, float(row[9]) * scale);
      }
#pragma unroll
      for (int ib = 0; ib < 2; ++ib) {
        const int local_m = warp_m * 32 + ib * 16;
        const __nv_bfloat16* a0p = &sa[buffer][(local_m + fragment_row) * kAS + kb + kk];
        const __nv_bfloat16* a1p = &sa[buffer][(local_m + fragment_row + 8) * kAS + kb + kk];
        const uint32_t a0 = *reinterpret_cast<const uint32_t*>(a0p);
        const uint32_t a1 = *reinterpret_cast<const uint32_t*>(a1p);
        const uint32_t a2 = *reinterpret_cast<const uint32_t*>(a0p + 8);
        const uint32_t a3 = *reinterpret_cast<const uint32_t*>(a1p + 8);
#pragma unroll
        for (int jb = 0; jb < 4; ++jb) {
          mma(accum[ib][jb][0], accum[ib][jb][1], accum[ib][jb][2], accum[ib][jb][3],
              a0, a1, a2, a3, b0[jb], b1[jb]);
        }
      }
    }
    __syncthreads();
    if (tile + 2 < tiles) {
      load(buffer, tile + 2);
      asm volatile("cp.async.commit_group;\n" ::);
    }
  }

  const int cr = lane >> 2;
  const int cc = (lane & 3) * 2;
#pragma unroll
  for (int ib = 0; ib < 2; ++ib) {
    const int row_base = bm + warp_m * 32 + ib * 16;
#pragma unroll
    for (int jb = 0; jb < 4; ++jb) {
      const int col = bn + warp_n * 32 + jb * 8 + cc;
      const int r0 = row_base + cr;
      const int r1 = r0 + 8;
      if (col < n && r0 < m) out[static_cast<size_t>(r0) * n + col] = __float2bfloat16(accum[ib][jb][0]);
      if (col < n && r1 < m) out[static_cast<size_t>(r1) * n + col] = __float2bfloat16(accum[ib][jb][2]);
      if (col + 1 < n && r0 < m) out[static_cast<size_t>(r0) * n + col + 1] = __float2bfloat16(accum[ib][jb][1]);
      if (col + 1 < n && r1 < m) out[static_cast<size_t>(r1) * n + col + 1] = __float2bfloat16(accum[ib][jb][3]);
    }
  }
}

template <int Rows, int Warps>
__global__ void w8a16_smallm_kernel(const __nv_bfloat16* x,
                                    const int8_t* weight,
                                    const float* scales,
                                    __nv_bfloat16* out,
                                    int n, int k) {
  const int lane = threadIdx.x & 31;
  const int row_n = blockIdx.x * Warps + (threadIdx.x >> 5);
  if (row_n >= n) return;
  float sum[Rows] = {};
  const int8_t* w = weight + static_cast<size_t>(row_n) * k;
  for (int col = lane * 4; col < k; col += 32 * 4) {
    const char4 values = *reinterpret_cast<const char4*>(w + col);
#pragma unroll
    for (int row_m = 0; row_m < Rows; ++row_m) {
      const auto* input = x + static_cast<size_t>(row_m) * k + col;
      const float2 x01 = __bfloat1622float2(
          *reinterpret_cast<const __nv_bfloat162*>(input));
      const float2 x23 = __bfloat1622float2(
          *reinterpret_cast<const __nv_bfloat162*>(input + 2));
      sum[row_m] = fmaf(float(values.x), x01.x, sum[row_m]);
      sum[row_m] = fmaf(float(values.y), x01.y, sum[row_m]);
      sum[row_m] = fmaf(float(values.z), x23.x, sum[row_m]);
      sum[row_m] = fmaf(float(values.w), x23.y, sum[row_m]);
    }
  }
#pragma unroll
  for (int row_m = 0; row_m < Rows; ++row_m) {
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
      sum[row_m] += __shfl_xor_sync(0xffffffff, sum[row_m], delta);
    }
    if (lane == 0) {
      out[static_cast<size_t>(row_m) * n + row_n] =
          __float2bfloat16(sum[row_m] * scales[row_n]);
    }
  }
}

template <int Rows, int Warps>
void launch_w8_smallm(const void* x, const void* quantized,
                      const void* scales, void* out, int n, int k,
                      cudaStream_t stream) {
  w8a16_smallm_kernel<Rows, Warps>
      <<<(n + Warps - 1) / Warps, Warps * 32, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x),
      static_cast<const int8_t*>(quantized),
      static_cast<const float*>(scales), static_cast<__nv_bfloat16*>(out),
      n, k);
}

__global__ void quantize_w8_kernel(const __nv_bfloat16* weight,
                                   int8_t* quantized, float* scales,
                                   int rows, int cols) {
  const int row = blockIdx.x;
  float local_max = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    local_max = fmaxf(local_max, fabsf(__bfloat162float(weight[static_cast<size_t>(row) * cols + col])));
  }
  for (int delta = 16; delta > 0; delta >>= 1) local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, delta));
  __shared__ float warp_max[8];
  if ((threadIdx.x & 31) == 0) warp_max[threadIdx.x >> 5] = local_max;
  __syncthreads();
  float amax = threadIdx.x < 8 ? warp_max[threadIdx.x] : 0.0f;
  if (threadIdx.x < 32) {
    for (int delta = 16; delta > 0; delta >>= 1) amax = fmaxf(amax, __shfl_down_sync(0xffffffff, amax, delta));
    if (threadIdx.x == 0) warp_max[0] = amax;
  }
  __syncthreads();
  const float scale = fmaxf(warp_max[0] / 127.0f, 1.0e-12f);
  if (threadIdx.x == 0) scales[row] = scale;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = __bfloat162float(weight[static_cast<size_t>(row) * cols + col]) / scale;
    quantized[static_cast<size_t>(row) * cols + col] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(value))));
  }
}

__global__ void dequantize_w8_kernel(const int8_t* quantized,
                                     const float* scales,
                                     __nv_bfloat16* weight,
                                     int rows, int cols) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rows * cols) return;
  const int row = idx / cols;
  weight[idx] = __float2bfloat16(float(quantized[idx]) * scales[row]);
}

}  // namespace

int quantize_w8_weight_bf16(const void* weight, void* quantized, void* scales,
                            int rows, int cols, cudaStream_t stream) {
  if (!weight || !quantized || !scales || rows <= 0 || cols <= 0 || cols % 64 != 0) return 1;
  quantize_w8_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(weight), static_cast<int8_t*>(quantized),
      static_cast<float*>(scales), rows, cols);
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? 0 : -static_cast<int>(error);
}

int dequantize_w8_weight_bf16(const void* quantized, const void* scales,
                              void* weight, int rows, int cols,
                              cudaStream_t stream) {
  if (!quantized || !scales || !weight || rows <= 0 || cols <= 0) return 1;
  const int total = rows * cols;
  dequantize_w8_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
      static_cast<const int8_t*>(quantized), static_cast<const float*>(scales),
      static_cast<__nv_bfloat16*>(weight), rows, cols);
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? 0 : -static_cast<int>(error);
}

int w8a16_linear_bf16(const void* x, const void* quantized,
                      const void* scales, void* out, int m, int n, int k,
                      int variant, cudaStream_t stream) {
  if (!x || !quantized || !scales || !out || m <= 0 || n <= 0 || k <= 0 || k % 64 != 0) return 1;
  if (variant < 0 || variant > 3) return 2;
  if (variant == 0 && m > 4) return 3;
  const bool matvec = variant >= 2 || variant == 0;
  if (matvec) {
    if (m > 8) return 3;
#define LAUNCH_ROWS(R) \
    do { \
      if (variant == 2 || (variant == 0 && m >= 2)) launch_w8_smallm<R, 4>(x, quantized, scales, out, n, k, stream); \
      else launch_w8_smallm<R, 8>(x, quantized, scales, out, n, k, stream); \
    } while (0)
    switch (m) {
      case 1: LAUNCH_ROWS(1); break;
      case 2: LAUNCH_ROWS(2); break;
      case 3: LAUNCH_ROWS(3); break;
      case 4: LAUNCH_ROWS(4); break;
      case 5: LAUNCH_ROWS(5); break;
      case 6: LAUNCH_ROWS(6); break;
      case 7: LAUNCH_ROWS(7); break;
      case 8: LAUNCH_ROWS(8); break;
    }
#undef LAUNCH_ROWS
  } else {
    w8a16_gemm_kernel<<<dim3((n + kBN - 1) / kBN, (m + kBM - 1) / kBM), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x), static_cast<const int8_t*>(quantized),
        static_cast<const float*>(scales), static_cast<__nv_bfloat16*>(out), m, n, k);
  }
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? 0 : -static_cast<int>(error);
}

}  // namespace flashrt_weight_only
