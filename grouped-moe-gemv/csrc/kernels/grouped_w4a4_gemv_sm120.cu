// SPDX-License-Identifier: Apache-2.0
//
// Grouped NVFP4 W4A4 expert projection for sm_120. See header.
//
// The block-scaled mma + swizzled-SF decode helpers below are copied
// verbatim from fp4_w4a4_mma_sm120.cu (file-local anonymous namespace, so
// they cannot be shared without modifying that validated file). The grouped
// kernel body derives token and expert pointers from a token-major device
// routing tensor. Correctness is pinned against dequantized mathematics and
// a bitwise per-route native loop in the package test suite.

#include "kernels/grouped_w4a4_gemv_sm120.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

#include "cute/arch/mma_sm120.hpp"
#include "cutlass/numeric_types.h"

namespace flash_rt {
namespace gemm {

namespace {

using AtomType = cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
    cutlass::float_e2m1_t,
    cutlass::float_e2m1_t,
    float,
    cutlass::float_ue4m3_t,
    16>;

constexpr int G_COLS_PER_WARP = 8;

__device__ __forceinline__ float decode_ue4m3(uint8_t raw) {
  int exponent = (raw >> 3) & 0xf;
  int mantissa = raw & 0x7;
  return exponent == 0
      ? ldexpf(static_cast<float>(mantissa), -9)
      : ldexpf(1.0f + static_cast<float>(mantissa) / 8.0f, exponent - 7);
}

__device__ __forceinline__ float fp4_block_dot(uint64_t a, uint64_t b) {
  float acc = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    auto av = static_cast<__nv_fp4x2_storage_t>(a >> (i * 8));
    auto bv = static_cast<__nv_fp4x2_storage_t>(b >> (i * 8));
    __half2_raw ar = __nv_cvt_fp4x2_to_halfraw2(av, __NV_E2M1);
    __half2_raw br = __nv_cvt_fp4x2_to_halfraw2(bv, __NV_E2M1);
    float2 af = __half22float2(*reinterpret_cast<const __half2*>(&ar));
    float2 bf = __half22float2(*reinterpret_cast<const __half2*>(&br));
    acc = fmaf(af.x, bf.x, acc);
    acc = fmaf(af.y, bf.y, acc);
  }
  return acc;
}

__device__ __forceinline__ int sf_offset(int row, int k_block,
                                         int n_col_super) {
  int rb = row >> 7;
  int ri = row & 127;
  return (rb * n_col_super + (k_block >> 2)) * 512
      + (ri & 31) * 16 + ((ri >> 5) & 3) * 4 + (k_block & 3);
}

// Contract fallback for K divisible by 16 but not 64. Target decode/verify
// shapes use the MMA path below; this path keeps the public K%16 boundary real.
__global__ void grouped_simt_kernel(
    const uint8_t* __restrict__ A_all,
    const uint8_t* __restrict__ B_stack,
    const uint8_t* __restrict__ SFA,
    const uint8_t* __restrict__ SFB_stack,
    __nv_bfloat16* __restrict__ D,
    const float* __restrict__ alpha_stack,
    const int* __restrict__ expert_idx,
    int N, int K, int top_k, long w_stride, long sfb_stride) {
  int slot = blockIdx.y;
  int token = slot / top_k;
  int expert = expert_idx[slot];
  int lane = threadIdx.x & 31;
  int row = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (row >= N) return;

  const uint64_t* a = reinterpret_cast<const uint64_t*>(
      A_all + static_cast<long>(token) * (K / 2));
  const uint64_t* b = reinterpret_cast<const uint64_t*>(
      B_stack + static_cast<long>(expert) * w_stride
      + static_cast<long>(row) * (K / 2));
  const uint8_t* sfb = SFB_stack + static_cast<long>(expert) * sfb_stride;
  int n_col_super = ((K / 16) + 3) / 4;
  float acc = 0.0f;
  for (int kb = lane; kb < K / 16; kb += 32) {
    float sa = decode_ue4m3(SFA[sf_offset(token, kb, n_col_super)]);
    float sb = decode_ue4m3(sfb[sf_offset(row, kb, n_col_super)]);
    acc += fp4_block_dot(a[kb], b[kb]) * sa * sb;
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    acc += __shfl_xor_sync(0xffffffff, acc, offset);
  if (lane == 0) {
    D[static_cast<long>(slot) * N + row] =
        __float2bfloat16(acc * alpha_stack[expert]);
  }
}

__device__ __forceinline__ uint32_t fast_load_a(
    const uint8_t* sA, int t0, int t1, int reg_idx) {
  int row_off = ((reg_idx & 1) ? (t1 + 8) : t1) * 32;
  int col_off = t0 * 4 + ((reg_idx >> 1) & 1) * 16;
  return *reinterpret_cast<const uint32_t*>(sA + row_off + col_off);
}

__device__ __forceinline__ uint32_t fast_load_b(
    const uint8_t* sB, int t0, int t1, int reg_idx) {
  int col_off = t0 * 4 + reg_idx * 16;
  return *reinterpret_cast<const uint32_t*>(sB + t1 * 32 + col_off);
}

__device__ __forceinline__ uint32_t fast_load_sfa(
    const uint8_t* sSFA, int unique_row) {
  return *reinterpret_cast<const uint32_t*>(sSFA + unique_row * 4);
}

__device__ __forceinline__ uint32_t fast_load_sfb(
    const uint8_t* sSFB, int unique_col) {
  return *reinterpret_cast<const uint32_t*>(sSFB + unique_col * 4);
}

__device__ __forceinline__ void cp_async_4(
    uint8_t* smem_dst, const uint8_t* gmem_src) {
  uint32_t smem_int = __cvta_generic_to_shared(smem_dst);
  asm volatile(
      "cp.async.ca.shared.global.L2::128B [%0], [%1], 4;\n"
      :: "r"(smem_int), "l"(gmem_src));
}

__device__ __forceinline__ void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_group(int N) {
  if (N == 0) {
    asm volatile("cp.async.wait_group 0;\n" ::);
  } else if (N == 1) {
    asm volatile("cp.async.wait_group 1;\n" ::);
  } else {
    asm volatile("cp.async.wait_all;\n" ::);
  }
}

// full_n_kernel body + token-major routed-pair indexing. Activations are
// quantized once per token and reused by every selected expert.
template <int WARPS>
__global__ void grouped_n_kernel(
    const uint8_t* __restrict__ A_packed_all,
    const uint8_t* __restrict__ B_stack,
    const uint8_t* __restrict__ SFA_all,
    const uint8_t* __restrict__ SFB_stack,
    __nv_bfloat16* __restrict__ D,
    const float* __restrict__ alpha_stack,
    const int* __restrict__ expert_idx,
    int N, int K, int top_k,
    long w_stride, long sfb_stride) {
  // expert_idx is flattened token-major: [M, top_k].
  int slot = blockIdx.y;
  int token = slot / top_k;
  int e = expert_idx[slot];
  const uint8_t* A_packed = A_packed_all + static_cast<long>(token) * (K / 2);
  const uint8_t* B_packed = B_stack + static_cast<long>(e) * w_stride;
  const uint8_t* SFB = SFB_stack + static_cast<long>(e) * sfb_stride;
  __nv_bfloat16* D_e = D + static_cast<long>(slot) * N;
  float alpha = alpha_stack[e];

  __shared__ alignas(16) uint8_t s_A[2][WARPS][16 * 32];
  __shared__ alignas(16) uint8_t s_SFA[2][WARPS][16 * 4];
  __shared__ alignas(16) uint8_t s_B_all[2][WARPS * 8 * 32];
  __shared__ alignas(16) uint8_t s_SFB_all[2][WARPS * 8 * 4];

  int tid = threadIdx.x;
  int warp = tid >> 5;
  int lane = tid & 31;

  int block_n_off = blockIdx.x * WARPS * G_COLS_PER_WARP;
  int my_n_off = block_n_off + warp * G_COLS_PER_WARP;
  if (my_n_off >= N) return;

  int t0 = lane & 3;
  int t1 = lane >> 2;
  int sfa_unique_row = (lane & 1) * 8 + (lane >> 2);
  int sfb_unique_col = lane >> 2;

  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;

  const int K_iters = K / 64;
  const int K_half = K / 2;

  if (lane < 16) {
    int row = lane;
    if (row >= 1 && row <= 15) {
      int4* a0_v = reinterpret_cast<int4*>(s_A[0][warp]);
      int4* a1_v = reinterpret_cast<int4*>(s_A[1][warp]);
      int4 z; z.x = 0; z.y = 0; z.z = 0; z.w = 0;
      a0_v[row * 2 + 0] = z; a0_v[row * 2 + 1] = z;
      a1_v[row * 2 + 0] = z; a1_v[row * 2 + 1] = z;
    }
  }
  if (lane < 4) {
    for (int i = 4 + lane; i < 64; i += 4) {
      s_SFA[0][warp][i] = 0;
      s_SFA[1][warp][i] = 0;
    }
  }

  const int K_blocks = K / 16;
  const int n_col_super = (K_blocks + 3) / 4;

  auto issue_async_load = [&](int buf, int kt) {
    int byte_off = kt * 32;
    if (lane < 8) {
      cp_async_4(s_A[buf][warp] + lane * 4, A_packed + byte_off + lane * 4);
    }
    if (lane == 0) {
      // CUTLASS batched SFA layout. This is the same addressing used by the
      // M<=16 warp-split kernel and lets one [M,K] quantization feed M*top_k
      // expert projections without materializing repeated activations.
      int rb = token >> 7;
      int ri = token & 127;
      int super_idx = rb * n_col_super + kt;
      int inner_base = (ri & 31) * 16 + ((ri >> 5) & 3) * 4;
      cp_async_4(s_SFA[buf][warp] + 0, SFA_all + super_idx * 512 + inner_base);
    }
    {
      uint8_t* my_s_B = s_B_all[buf] + warp * (8 * 32);
      for (int c = 0; c < 2; ++c) {
        int chunk = lane + c * 32;
        int col = chunk >> 3;
        int off = chunk & 7;
        cp_async_4(
            my_s_B + chunk * 4,
            B_packed + (my_n_off + col) * K_half + byte_off + off * 4);
      }
    }
    if (lane < 8) {
      uint8_t* my_s_SFB = s_SFB_all[buf] + warp * (8 * 4);
      int col = my_n_off + lane;
      int rb = col >> 7;
      int ri = col & 127;
      int super_idx = rb * n_col_super + kt;
      int inner_base = (ri & 31) * 16 + ((ri >> 5) & 3) * 4;
      cp_async_4(
          my_s_SFB + lane * 4,
          SFB + super_idx * 512 + inner_base);
    }
  };

  issue_async_load(0, 0);
  cp_async_commit_group();
  if (K_iters > 1) {
    issue_async_load(1, 1);
    cp_async_commit_group();
  }

  for (int kt = 0; kt < K_iters; ++kt) {
    int curr_buf = kt & 1;

    if (kt + 1 < K_iters) {
      cp_async_wait_group(1);
    } else {
      cp_async_wait_group(0);
    }
    __syncwarp();

    uint32_t a0 = fast_load_a(s_A[curr_buf][warp], t0, t1, 0);
    uint32_t a1 = fast_load_a(s_A[curr_buf][warp], t0, t1, 1);
    uint32_t a2 = fast_load_a(s_A[curr_buf][warp], t0, t1, 2);
    uint32_t a3 = fast_load_a(s_A[curr_buf][warp], t0, t1, 3);
    uint8_t* my_s_B = s_B_all[curr_buf] + warp * (8 * 32);
    uint8_t* my_s_SFB = s_SFB_all[curr_buf] + warp * (8 * 4);
    uint32_t b0 = fast_load_b(my_s_B, t0, t1, 0);
    uint32_t b1 = fast_load_b(my_s_B, t0, t1, 1);
    uint32_t sfa = fast_load_sfa(s_SFA[curr_buf][warp], sfa_unique_row);
    uint32_t sfb = fast_load_sfb(my_s_SFB, sfb_unique_col);

    float d0, d1, d2, d3;
    AtomType::fma(d0, d1, d2, d3,
                  a0, a1, a2, a3,
                  b0, b1,
                  c0, c1, c2, c3,
                  sfa, sfb);
    c0 = d0; c1 = d1; c2 = d2; c3 = d3;

    if (kt + 2 < K_iters) {
      issue_async_load(curr_buf, kt + 2);
      cp_async_commit_group();
    }
  }

  int q = lane >> 2;
  int r = lane & 3;
  if (q == 0) {
    int col0 = my_n_off + r * 2;
    int col1 = col0 + 1;
    if (col0 < N) D_e[col0] = __float2bfloat16(c0 * alpha);
    if (col1 < N) D_e[col1] = __float2bfloat16(c1 * alpha);
  }
}

}  // namespace

int grouped_w4a4_gemv_sm120_bf16_impl(
    const void*  A_packed,
    const void*  B_stack,
    void*        D,
    const void*  SFA,
    const void*  SFB_stack,
    const void*  alpha_stack,
    const void*  expert_idx,
    int          M,
    int          top_k,
    int          N,
    int          K,
    long         w_stride,
    long         sfb_stride,
    cudaStream_t stream) {
  if (!A_packed || !B_stack || !D || !SFA || !SFB_stack ||
      !alpha_stack || !expert_idx) return 1;
  if (K <= 0 || (K % 16) != 0) return 2;
  if (N <= 0 || (N % G_COLS_PER_WARP) != 0) return 3;
  if (M <= 0 || top_k <= 0) return 4;

  const int pairs = M * top_k;
  if ((K % 64) != 0 || K > 512 || pairs <= 8) {
    dim3 block(256);
    dim3 grid((N + 7) / 8, M * top_k);
    grouped_simt_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(A_packed),
        reinterpret_cast<const uint8_t*>(B_stack),
        reinterpret_cast<const uint8_t*>(SFA),
        reinterpret_cast<const uint8_t*>(SFB_stack),
        reinterpret_cast<__nv_bfloat16*>(D),
        reinterpret_cast<const float*>(alpha_stack),
        reinterpret_cast<const int*>(expert_idx), N, K, top_k, w_stride,
        sfb_stride);
    return 0;
  }

  const int warps = K <= 512 ? 8 : 4;
  dim3 block(warps * 32);
  dim3 grid((N + warps * G_COLS_PER_WARP - 1) /
                (warps * G_COLS_PER_WARP),
            M * top_k);
#define LAUNCH_GROUPED(WARPS)                                                   \
  grouped_n_kernel<WARPS><<<grid, block, 0, stream>>>(                         \
      reinterpret_cast<const uint8_t*>(A_packed),                              \
      reinterpret_cast<const uint8_t*>(B_stack),                               \
      reinterpret_cast<const uint8_t*>(SFA),                                   \
      reinterpret_cast<const uint8_t*>(SFB_stack),                             \
      reinterpret_cast<__nv_bfloat16*>(D),                                     \
      reinterpret_cast<const float*>(alpha_stack),                             \
      reinterpret_cast<const int*>(expert_idx), N, K, top_k, w_stride,         \
      sfb_stride)
  if (warps == 8) {
    LAUNCH_GROUPED(8);
  } else {
    LAUNCH_GROUPED(4);
  }
#undef LAUNCH_GROUPED
  return 0;
}

}  // namespace gemm
}  // namespace flash_rt
