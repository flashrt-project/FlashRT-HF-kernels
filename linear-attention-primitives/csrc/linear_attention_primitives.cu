// SPDX-License-Identifier: Apache-2.0

#include "linear_attention_primitives.cuh"

namespace flash_rt::linear_attention_primitives {
namespace {

constexpr int kWarpsPerBlock = 8;
constexpr int kThreads = kWarpsPerBlock * 32;

template<int K_FIXED>
__global__ void bf16_matvec_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ out,
    int n) {
  __shared__ __nv_bfloat16 x_sh[K_FIXED];
  const int4* x_i4 = reinterpret_cast<const int4*>(x);
  int4* x_sh_i4 = reinterpret_cast<int4*>(x_sh);
  const int k_i4 = K_FIXED / 8;
  for (int j = threadIdx.x; j < k_i4; j += kThreads) {
    x_sh_i4[j] = x_i4[j];
  }
  __syncthreads();

  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * kWarpsPerBlock + warp_id;
  if (row >= n) return;

  const int4* w_i4 = reinterpret_cast<const int4*>(w + row * K_FIXED);
  float acc = 0.0f;
  for (int i4 = lane; i4 < k_i4; i4 += 32) {
    int4 wv = w_i4[i4];
    int4 xv = x_sh_i4[i4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      __nv_bfloat162 wb = *reinterpret_cast<__nv_bfloat162*>(
          &(reinterpret_cast<int*>(&wv)[i]));
      __nv_bfloat162 xb = *reinterpret_cast<__nv_bfloat162*>(
          &(reinterpret_cast<int*>(&xv)[i]));
      float2 wf = __bfloat1622float2(wb);
      float2 xf = __bfloat1622float2(xb);
      acc = fmaf(xf.x, wf.x, acc);
      acc = fmaf(xf.y, wf.y, acc);
    }
  }

  #pragma unroll
  for (int off = 16; off > 0; off /= 2) {
    acc += __shfl_xor_sync(0xffffffff, acc, off);
  }
  if (lane == 0) out[row] = __float2bfloat16(acc);
}

__global__ void bf16_matvec_generic_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ out,
    int n,
    int k) {
  extern __shared__ __nv_bfloat16 x_sh[];
  constexpr int kChunk = 4096;
  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * kWarpsPerBlock + warp_id;
  float acc = 0.0f;

  for (int k0 = 0; k0 < k; k0 += kChunk) {
    const int chunk = min(kChunk, k - k0);
    for (int j = threadIdx.x; j < chunk; j += kThreads) {
      x_sh[j] = x[k0 + j];
    }
    __syncthreads();
    if (row < n) {
      const __nv_bfloat16* w_row = w + static_cast<size_t>(row) * k + k0;
      for (int j = lane; j < chunk; j += 32) {
        acc = fmaf(__bfloat162float(x_sh[j]), __bfloat162float(w_row[j]), acc);
      }
    }
    __syncthreads();
  }
  if (row >= n) return;
  #pragma unroll
  for (int off = 16; off > 0; off /= 2) {
    acc += __shfl_xor_sync(0xffffffff, acc, off);
  }
  if (lane == 0) out[row] = __float2bfloat16(acc);
}

template<int K_FIXED>
__global__ void bf16_smallm_matmul_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ out,
    int m,
    int n) {
  __shared__ __nv_bfloat16 x_sh[K_FIXED];
  const int row_m = blockIdx.y;
  if (row_m >= m) return;

  const int4* x_i4 = reinterpret_cast<const int4*>(x + static_cast<size_t>(row_m) * K_FIXED);
  int4* x_sh_i4 = reinterpret_cast<int4*>(x_sh);
  const int k_i4 = K_FIXED / 8;
  for (int j = threadIdx.x; j < k_i4; j += kThreads) {
    x_sh_i4[j] = x_i4[j];
  }
  __syncthreads();

  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const int row_n = blockIdx.x * kWarpsPerBlock + warp_id;
  if (row_n >= n) return;

  const int4* w_i4 = reinterpret_cast<const int4*>(w + static_cast<size_t>(row_n) * K_FIXED);
  float acc = 0.0f;
  for (int i4 = lane; i4 < k_i4; i4 += 32) {
    int4 wv = w_i4[i4];
    int4 xv = x_sh_i4[i4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      __nv_bfloat162 wb = *reinterpret_cast<__nv_bfloat162*>(
          &(reinterpret_cast<int*>(&wv)[i]));
      __nv_bfloat162 xb = *reinterpret_cast<__nv_bfloat162*>(
          &(reinterpret_cast<int*>(&xv)[i]));
      float2 wf = __bfloat1622float2(wb);
      float2 xf = __bfloat1622float2(xb);
      acc = fmaf(xf.x, wf.x, acc);
      acc = fmaf(xf.y, wf.y, acc);
    }
  }
  #pragma unroll
  for (int off = 16; off > 0; off /= 2) {
    acc += __shfl_xor_sync(0xffffffff, acc, off);
  }
  if (lane == 0) out[static_cast<size_t>(row_m) * n + row_n] = __float2bfloat16(acc);
}

__global__ void bf16_smallm_matmul_generic_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ out,
    int m,
    int n,
    int k) {
  extern __shared__ __nv_bfloat16 x_sh[];
  constexpr int kChunk = 4096;
  const int row_m = blockIdx.y;
  if (row_m >= m) return;
  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const int row_n = blockIdx.x * kWarpsPerBlock + warp_id;
  float acc = 0.0f;

  for (int k0 = 0; k0 < k; k0 += kChunk) {
    const int chunk = min(kChunk, k - k0);
    for (int j = threadIdx.x; j < chunk; j += kThreads) {
      x_sh[j] = x[static_cast<size_t>(row_m) * k + k0 + j];
    }
    __syncthreads();
    if (row_n < n) {
      const __nv_bfloat16* w_row = w + static_cast<size_t>(row_n) * k + k0;
      for (int j = lane; j < chunk; j += 32) {
        acc = fmaf(__bfloat162float(x_sh[j]), __bfloat162float(w_row[j]), acc);
      }
    }
    __syncthreads();
  }
  if (row_n >= n) return;
  #pragma unroll
  for (int off = 16; off > 0; off /= 2) {
    acc += __shfl_xor_sync(0xffffffff, acc, off);
  }
  if (lane == 0) out[static_cast<size_t>(row_m) * n + row_n] = __float2bfloat16(acc);
}

template<int M_TILE>
__global__ void bf16_ab96_mtile_pair_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ out,
    int m) {
  constexpr int K_FIXED = 5120;
  extern __shared__ __align__(16) __nv_bfloat16 x_sh[];
  const int m0 = blockIdx.y * M_TILE;
  const int k_i4 = K_FIXED / 8;
  const int4* x_i4 = reinterpret_cast<const int4*>(x);
  int4* x_sh_i4 = reinterpret_cast<int4*>(x_sh);
  const int x_i4_total = M_TILE * k_i4;
  for (int j = threadIdx.x; j < x_i4_total; j += kThreads) {
    const int mt = j / k_i4;
    const int ki4 = j - mt * k_i4;
    const int row_m = m0 + mt;
    x_sh_i4[j] = (row_m < m) ? x_i4[row_m * k_i4 + ki4] : make_int4(0, 0, 0, 0);
  }
  __syncthreads();

  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const int n0 = blockIdx.x * (kWarpsPerBlock * 2) + warp_id * 2;
  const int n1 = n0 + 1;
  if (n0 >= 96) return;

  const int4* w0_i4 = reinterpret_cast<const int4*>(w + n0 * K_FIXED);
  const int4* w1_i4 = reinterpret_cast<const int4*>(w + n1 * K_FIXED);
  float acc0[M_TILE];
  float acc1[M_TILE];
  #pragma unroll
  for (int mt = 0; mt < M_TILE; ++mt) {
    acc0[mt] = 0.0f;
    acc1[mt] = 0.0f;
  }

  for (int i4 = lane; i4 < k_i4; i4 += 32) {
    int4 w0v = w0_i4[i4];
    int4 w1v = w1_i4[i4];
    int4 xv[M_TILE];
    #pragma unroll
    for (int mt = 0; mt < M_TILE; ++mt) xv[mt] = x_sh_i4[mt * k_i4 + i4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      __nv_bfloat162 w0b = *reinterpret_cast<__nv_bfloat162*>(
          &(reinterpret_cast<int*>(&w0v)[i]));
      __nv_bfloat162 w1b = *reinterpret_cast<__nv_bfloat162*>(
          &(reinterpret_cast<int*>(&w1v)[i]));
      float2 w0f = __bfloat1622float2(w0b);
      float2 w1f = __bfloat1622float2(w1b);
      #pragma unroll
      for (int mt = 0; mt < M_TILE; ++mt) {
        __nv_bfloat162 xb = *reinterpret_cast<__nv_bfloat162*>(
            &(reinterpret_cast<int*>(&xv[mt])[i]));
        float2 xf = __bfloat1622float2(xb);
        acc0[mt] = fmaf(xf.x, w0f.x, acc0[mt]);
        acc0[mt] = fmaf(xf.y, w0f.y, acc0[mt]);
        acc1[mt] = fmaf(xf.x, w1f.x, acc1[mt]);
        acc1[mt] = fmaf(xf.y, w1f.y, acc1[mt]);
      }
    }
  }

  #pragma unroll
  for (int off = 16; off > 0; off /= 2) {
    #pragma unroll
    for (int mt = 0; mt < M_TILE; ++mt) {
      acc0[mt] += __shfl_xor_sync(0xffffffff, acc0[mt], off);
      acc1[mt] += __shfl_xor_sync(0xffffffff, acc1[mt], off);
    }
  }
  if (lane == 0) {
    #pragma unroll
    for (int mt = 0; mt < M_TILE; ++mt) {
      const int row_m = m0 + mt;
      if (row_m < m) {
        out[row_m * 96 + n0] = __float2bfloat16(acc0[mt]);
        if (n1 < 96) out[row_m * 96 + n1] = __float2bfloat16(acc1[mt]);
      }
    }
  }
}

__global__ void split_qkv_gqa_kernel(
    const __nv_bfloat16* __restrict__ packed,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ v,
    int rows,
    int q_heads,
    int kv_heads,
    int head_dim) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int q_elems = rows * q_heads * head_dim;
  const int k_elems = rows * kv_heads * head_dim;
  const int v_elems = rows * kv_heads * head_dim;
  const int total = q_elems + k_elems + v_elems;
  if (idx >= total) return;
  if (idx < q_elems) {
    const int d = idx % head_dim;
    const int h = (idx / head_dim) % q_heads;
    const int r = idx / (q_heads * head_dim);
    const size_t src = static_cast<size_t>(r) * (q_heads + 2 * kv_heads) * head_dim + h * head_dim + d;
    q[idx] = packed[src];
  } else if (idx < q_elems + k_elems) {
    const int out = idx - q_elems;
    const int d = out % head_dim;
    const int h = (out / head_dim) % kv_heads;
    const int r = out / (kv_heads * head_dim);
    const size_t src = static_cast<size_t>(r) * (q_heads + 2 * kv_heads) * head_dim
        + (q_heads + h) * head_dim + d;
    k[out] = packed[src];
  } else {
    const int out = idx - q_elems - k_elems;
    const int d = out % head_dim;
    const int h = (out / head_dim) % kv_heads;
    const int r = out / (kv_heads * head_dim);
    const size_t src = static_cast<size_t>(r) * (q_heads + 2 * kv_heads) * head_dim
        + (q_heads + kv_heads + h) * head_dim + d;
    v[out] = packed[src];
  }
}

__global__ void split_qkv_broadcast_kernel(
    const __nv_bfloat16* __restrict__ packed,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ v,
    int rows,
    int q_heads,
    int kv_heads,
    int v_heads,
    int head_dim) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * v_heads * head_dim;
  if (idx >= total) return;
  const int d = idx % head_dim;
  const int h = (idx / head_dim) % v_heads;
  const int r = idx / (v_heads * head_dim);
  const int q_src_h = min(q_heads - 1, h * q_heads / v_heads);
  const int k_src_h = min(kv_heads - 1, h * kv_heads / v_heads);
  const size_t row = static_cast<size_t>(r) * (q_heads + kv_heads + v_heads) * head_dim;
  q[idx] = packed[row + q_src_h * head_dim + d];
  k[idx] = packed[row + (q_heads + k_src_h) * head_dim + d];
  v[idx] = packed[row + (q_heads + kv_heads) * head_dim + h * head_dim + d];
}

__device__ __forceinline__ __nv_bfloat16 rope_value(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    int row,
    int head,
    int col,
    int heads,
    int head_dim,
    int rope_dim) {
  const size_t base = (static_cast<size_t>(row) * heads + head) * head_dim;
  if (col >= rope_dim) return x[base + col];
  const int half = rope_dim >> 1;
  const int rot_col = (col < half) ? (col + half) : (col - half);
  float rot = __bfloat162float(x[base + rot_col]);
  if (col < half) rot = -rot;
  const float xv = __bfloat162float(x[base + col]);
  const float cv = __bfloat162float(cos[row * rope_dim + col]);
  const float sv = __bfloat162float(sin[row * rope_dim + col]);
  const float rot_sin_bf = __bfloat162float(__float2bfloat16(rot * sv));
  return __float2bfloat16(rot_sin_bf + xv * cv);
}

__global__ void partial_rope_qk_kernel(
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    int rows,
    int q_heads,
    int k_heads,
    int head_dim,
    int rope_dim) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int q_total = rows * q_heads * head_dim;
  const int k_total = rows * k_heads * head_dim;
  const int total = q_total + k_total;
  if (idx >= total) return;
  if (idx < q_total) {
    const int col = idx % head_dim;
    const int head = (idx / head_dim) % q_heads;
    const int row = idx / (head_dim * q_heads);
    q_out[idx] = rope_value(q_in, cos, sin, row, head, col, q_heads, head_dim, rope_dim);
  } else {
    const int k_idx = idx - q_total;
    const int col = k_idx % head_dim;
    const int head = (k_idx / head_dim) % k_heads;
    const int row = k_idx / (head_dim * k_heads);
    k_out[k_idx] = rope_value(k_in, cos, sin, row, head, col, k_heads, head_dim, rope_dim);
  }
}

__global__ void gated_delta_prepare_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    const float* __restrict__ neg_exp_a_log,
    const float* __restrict__ dt_bias,
    __nv_bfloat16* __restrict__ g_out,
    __nv_bfloat16* __restrict__ beta_out,
    int rows,
    int heads,
    int a_stride,
    int b_stride) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * heads;
  if (idx >= total) return;
  const int row = idx / heads;
  const int h = idx - row * heads;
  const float av = __bfloat162float(a[row * a_stride + h]) + dt_bias[h];
  const float sp = log1pf(expf(av));
  const float gv = neg_exp_a_log[h] * sp;
  const float bv = __bfloat162float(b[row * b_stride + h]);
  const float beta = 1.0f / (1.0f + expf(-bv));
  g_out[idx] = __float2bfloat16(gv);
  beta_out[idx] = __float2bfloat16(beta);
}

}  // namespace

void bf16_matvec(
    const __nv_bfloat16* x,
    const __nv_bfloat16* w,
    __nv_bfloat16* out,
    int n,
    int k,
    cudaStream_t stream) {
  const int grid = (n + kWarpsPerBlock - 1) / kWarpsPerBlock;
  if (k == 5120) {
    bf16_matvec_kernel<5120><<<grid, kThreads, 0, stream>>>(x, w, out, n);
  } else if (k == 4096) {
    bf16_matvec_kernel<4096><<<grid, kThreads, 0, stream>>>(x, w, out, n);
  } else {
    bf16_matvec_generic_kernel<<<grid, kThreads, 4096 * sizeof(__nv_bfloat16), stream>>>(
        x, w, out, n, k);
  }
}

void bf16_smallm_matmul(
    const __nv_bfloat16* x,
    const __nv_bfloat16* w,
    __nv_bfloat16* out,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  if (n == 96 && k == 5120) {
    constexpr int kMTile = 4;
    dim3 grid((96 + (kWarpsPerBlock * 2) - 1) / (kWarpsPerBlock * 2),
              (m + kMTile - 1) / kMTile);
    bf16_ab96_mtile_pair_kernel<kMTile>
        <<<grid, kThreads, kMTile * 5120 * sizeof(__nv_bfloat16), stream>>>(x, w, out, m);
    return;
  }
  dim3 grid((n + kWarpsPerBlock - 1) / kWarpsPerBlock, m);
  if (k == 5120) {
    bf16_smallm_matmul_kernel<5120><<<grid, kThreads, 0, stream>>>(x, w, out, m, n);
  } else if (k == 4096) {
    bf16_smallm_matmul_kernel<4096><<<grid, kThreads, 0, stream>>>(x, w, out, m, n);
  } else {
    bf16_smallm_matmul_generic_kernel<<<grid, kThreads, 4096 * sizeof(__nv_bfloat16), stream>>>(
        x, w, out, m, n, k);
  }
}

void split_qkv_gqa_bf16(
    const __nv_bfloat16* packed,
    __nv_bfloat16* q,
    __nv_bfloat16* k,
    __nv_bfloat16* v,
    int rows,
    int q_heads,
    int kv_heads,
    int head_dim,
    cudaStream_t stream) {
  const int total = rows * (q_heads + 2 * kv_heads) * head_dim;
  split_qkv_gqa_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
      packed, q, k, v, rows, q_heads, kv_heads, head_dim);
}

void split_qkv_broadcast_bf16(
    const __nv_bfloat16* packed,
    __nv_bfloat16* q,
    __nv_bfloat16* k,
    __nv_bfloat16* v,
    int rows,
    int q_heads,
    int kv_heads,
    int v_heads,
    int head_dim,
    cudaStream_t stream) {
  const int total = rows * v_heads * head_dim;
  split_qkv_broadcast_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
      packed, q, k, v, rows, q_heads, kv_heads, v_heads, head_dim);
}

void partial_rope_qk_bf16(
    const __nv_bfloat16* q_in,
    const __nv_bfloat16* k_in,
    const __nv_bfloat16* cos,
    const __nv_bfloat16* sin,
    __nv_bfloat16* q_out,
    __nv_bfloat16* k_out,
    int rows,
    int q_heads,
    int k_heads,
    int head_dim,
    int rope_dim,
    cudaStream_t stream) {
  const int total = rows * (q_heads + k_heads) * head_dim;
  partial_rope_qk_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
      q_in, k_in, cos, sin, q_out, k_out, rows, q_heads, k_heads, head_dim, rope_dim);
}

void gated_delta_prepare_bf16(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    const float* neg_exp_a_log,
    const float* dt_bias,
    __nv_bfloat16* g_out,
    __nv_bfloat16* beta_out,
    int rows,
    int heads,
    int a_stride,
    int b_stride,
    cudaStream_t stream) {
  const int total = rows * heads;
  gated_delta_prepare_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
      a, b, neg_exp_a_log, dt_bias, g_out, beta_out, rows, heads, a_stride, b_stride);
}

}  // namespace flash_rt::linear_attention_primitives
