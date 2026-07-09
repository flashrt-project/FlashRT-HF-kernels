// SPDX-License-Identifier: Apache-2.0
//
// Streaming linear-CE forward for small N over a huge fp32 vocabulary head.
//
// The (V, H) weight is the traffic; it is read exactly once. Each CTA owns
// kVPer*32 vocab rows and keeps ALL N hidden rows resident (staged per
// H-chunk in shared memory), so the weight is never re-read per hidden-row
// tile. Staging is double-buffered with cp.async so the global-memory
// latency of the next chunk hides under the current chunk's math. With
// kVPer=2 each thread owns two vocab rows, so every hidden float4 pulled
// from shared memory feeds two FMA chains — the shared-memory read rate is
// what bounds the large-N cases, not DRAM. The epilogue writes the logits
// tile (needed by the backward), gathers the label logits, and emits one
// online-softmax partial (max, sumexp) per (row, 32-wide vocab tile) for a
// cheap downstream merge.

#include <cuda_pipeline_primitives.h>

#include "vocab_ce_train.cuh"

namespace flashrt_hub {
namespace vocab_ce_train {

namespace {

constexpr int kPad = 4;  // 16B row pad: float4-aligned and bank-staggered

template <int kThreadsT, int kVRows, int kHChunk>
__device__ __forceinline__ void stage_chunk(const float* __restrict__ hidden,
                                            const float* __restrict__ weight,
                                            float* __restrict__ h_buf,
                                            float* __restrict__ w_buf, int v0,
                                            int h0, int rows, int h) {
  constexpr int kStride = kHChunk + kPad;
  // hidden rows: rows x kHChunk floats, 16B copies
  for (int idx = threadIdx.x; idx < rows * (kHChunk / 4); idx += kThreadsT) {
    const int r = idx / (kHChunk / 4);
    const int c4 = idx - r * (kHChunk / 4);
    __pipeline_memcpy_async(h_buf + r * kStride + c4 * 4,
                            hidden + (long)r * h + h0 + c4 * 4, 16);
  }
  // weight tile: kVRows x kHChunk floats
  for (int idx = threadIdx.x; idx < kVRows * (kHChunk / 4); idx += kThreadsT) {
    const int r = idx / (kHChunk / 4);
    const int c4 = idx - r * (kHChunk / 4);
    __pipeline_memcpy_async(w_buf + r * kStride + c4 * 4,
                            weight + (long)(v0 + r) * h + h0 + c4 * 4, 16);
  }
}

template <int kThreadsT, int kVPer, int kJn, int kHChunk, int kStages>
__global__ void __launch_bounds__(kThreadsT) vocab_ce_fwd_kernel(
    const float* __restrict__ hidden, const float* __restrict__ weight,
    const long* __restrict__ labels, float* __restrict__ logits,
    float* __restrict__ partial_max, float* __restrict__ partial_sum,
    float* __restrict__ label_logit, int rows, int v_total, int h) {
  extern __shared__ float smem[];
  constexpr int kWarpsT = kThreadsT / 32;
  constexpr int kVRows = kVPer * 32;
  constexpr int kStride = kHChunk + kPad;
  const int h_buf_elems = rows * kStride;
  const int w_buf_elems = kVRows * kStride;
  // layout: h[0..kStages-1], w[0..kStages-1]
  float* h_bufs[kStages];
  float* w_bufs[kStages];
#pragma unroll
  for (int i = 0; i < kStages; ++i) {
    h_bufs[i] = smem + i * h_buf_elems;
    w_bufs[i] = smem + kStages * h_buf_elems + i * w_buf_elems;
  }

  const int v0 = blockIdx.x * kVRows;
  const int lane = threadIdx.x & 31;  // vocab row within the 32-wide subtile
  const int grp = threadIdx.x >> 5;   // hidden-row group (stride kWarpsT)
  const int p_stride = gridDim.x * kVPer;  // partial tiles are 32-wide
  const int num_chunks = h / kHChunk;

  float acc[kVPer][kJn];
  const int jn = (rows - grp + kWarpsT - 1) / kWarpsT;  // rows for this group
#pragma unroll
  for (int p = 0; p < kVPer; ++p)
#pragma unroll
    for (int j = 0; j < kJn; ++j) acc[p][j] = 0.0f;

  // prologue: fill kStages-1 buffers ahead
  for (int k = 0; k < kStages - 1 && k < num_chunks; ++k) {
    stage_chunk<kThreadsT, kVRows, kHChunk>(hidden, weight, h_bufs[k],
                                            w_bufs[k], v0, k * kHChunk, rows,
                                            h);
    __pipeline_commit();
  }

  for (int k = 0; k < num_chunks; ++k) {
    const int buf = k % kStages;
    const int ahead = k + kStages - 1;
    if (ahead < num_chunks) {
      stage_chunk<kThreadsT, kVRows, kHChunk>(hidden, weight,
                                              h_bufs[ahead % kStages],
                                              w_bufs[ahead % kStages], v0,
                                              ahead * kHChunk, rows, h);
      __pipeline_commit();
      __pipeline_wait_prior(kStages - 1);
    } else {
      __pipeline_wait_prior(num_chunks - 1 - k);
    }
    __syncthreads();

    const float4* wrow[kVPer];
#pragma unroll
    for (int p = 0; p < kVPer; ++p)
      wrow[p] = reinterpret_cast<const float4*>(w_bufs[buf] +
                                                (p * 32 + lane) * kStride);
    const float4* hrow =
        reinterpret_cast<const float4*>(h_bufs[buf] + grp * kStride);
    constexpr int kRowStride4 = kWarpsT * kStride / 4;
#pragma unroll 2
    for (int hh = 0; hh < kHChunk / 4; ++hh) {
      float4 wv[kVPer];
#pragma unroll
      for (int p = 0; p < kVPer; ++p) wv[p] = wrow[p][hh];
#pragma unroll
      for (int j = 0; j < kJn; ++j) {
        if (j < jn) {
          const float4 hv = hrow[j * kRowStride4 + hh];
#pragma unroll
          for (int p = 0; p < kVPer; ++p)
            acc[p][j] +=
                wv[p].x * hv.x + wv[p].y * hv.y + wv[p].z * hv.z + wv[p].w * hv.w;
        }
      }
    }
    __syncthreads();  // buf is re-staged kStages iterations from now
  }

  // epilogue: logits write, label gather, per-32-wide-tile softmax partials
  for (int j = 0; j < jn && j < kJn; ++j) {
    const int n = grp + j * kWarpsT;
#pragma unroll
    for (int p = 0; p < kVPer; ++p) {
      const float logit = acc[p][j];
      const int vcol = v0 + p * 32 + lane;
      logits[(long)n * v_total + vcol] = logit;
      if (labels[n] == (long)vcol) label_logit[n] = logit;

      float m = logit;
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
      float e = __expf(logit - m);
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        e += __shfl_xor_sync(0xffffffffu, e, off);
      if (lane == 0) {
        partial_max[(long)n * p_stride + blockIdx.x * kVPer + p] = m;
        partial_sum[(long)n * p_stride + blockIdx.x * kVPer + p] = e;
      }
    }
  }
}

template <int kVPer, int kHChunk, int kStages>
int smem_bytes_for(int rows) {
  return kStages * (rows + kVPer * 32) * (kHChunk + kPad) * (int)sizeof(float);
}

template <int kThreadsT, int kVPer, int kHChunk, int kStages>
void launch_all_jn(const float* hidden, const float* weight,
                   const long* labels, float* logits, float* partial_max,
                   float* partial_sum, float* label_logit, int rows, int v,
                   int h, cudaStream_t stream) {
  constexpr int kWarpsT = kThreadsT / 32;
  const int smem = smem_bytes_for<kVPer, kHChunk, kStages>(rows);
  const int jn = (rows + kWarpsT - 1) / kWarpsT;
  dim3 grid(v / (kVPer * 32)), block(kThreadsT);
#define CASE(J)                                                               \
  case J: {                                                                   \
    static bool configured_##J = false;                                       \
    if (!configured_##J) {                                                    \
      cudaFuncSetAttribute(                                                   \
          vocab_ce_fwd_kernel<kThreadsT, kVPer, J, kHChunk, kStages>,         \
          cudaFuncAttributeMaxDynamicSharedMemorySize,                        \
          smem_bytes_for<kVPer, kHChunk, kStages>(kMaxRows));                 \
      configured_##J = true;                                                  \
    }                                                                         \
    vocab_ce_fwd_kernel<kThreadsT, kVPer, J, kHChunk, kStages>               \
        <<<grid, block, smem, stream>>>(hidden, weight, labels, logits,       \
                                        partial_max, partial_sum,             \
                                        label_logit, rows, v, h);             \
    break;                                                                    \
  }
  switch (jn) {
    CASE(1) CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8)
    CASE(9) CASE(10) CASE(11) CASE(12) CASE(13) CASE(14) CASE(15) CASE(16)
    default:
      break;  // rows > kMaxRows rejected by the binding
  }
#undef CASE
}

}  // namespace

void vocab_ce_fwd_launch(const float* hidden, const float* weight,
                         const long* labels, float* logits, float* partial_max,
                         float* partial_sum, float* label_logit, int rows,
                         int v, int h, cudaStream_t stream) {
  // Adaptive tiling, swept on a 5090. N <= 64: 32-wide vocab tiles with
  // 64-float chunks keep two CTAs per SM resident. Above that each thread
  // takes two vocab rows (64-wide tiles) so every hidden float4 pulled from
  // shared memory feeds two FMA chains; past ~112 rows the staging buffers
  // exceed half the SM's shared memory, so a 512-thread block trades the
  // second CTA for more warps in the one that fits. Wider blocks (768+),
  // 16-float chunks, 4 vocab rows per thread and L2-direct hidden reads all
  // measured slower.
  if (rows <= 64 && h % 64 == 0) {
    launch_all_jn<256, 1, 64, 2>(hidden, weight, labels, logits, partial_max,
                                 partial_sum, label_logit, rows, v, h, stream);
  } else if (v % 64 != 0) {
    launch_all_jn<256, 1, 32, 2>(hidden, weight, labels, logits, partial_max,
                                 partial_sum, label_logit, rows, v, h, stream);
  } else if (rows <= 112) {
    launch_all_jn<256, 2, 32, 2>(hidden, weight, labels, logits, partial_max,
                                 partial_sum, label_logit, rows, v, h, stream);
  } else {
    launch_all_jn<512, 2, 32, 2>(hidden, weight, labels, logits, partial_max,
                                 partial_sum, label_logit, rows, v, h, stream);
  }
}

namespace {

constexpr int kStatsThreads = 256;

// grid (rows, kStatsSplits); each CTA owns a contiguous V-slice of one
// logits row and reduces it to one online-softmax partial. Reads are
// float4-coalesced; V is a multiple of 32 so the fp32 tail loop is short.
__global__ void __launch_bounds__(kStatsThreads) vocab_ce_stats_kernel(
    const float* __restrict__ logits, float* __restrict__ partial_max,
    float* __restrict__ partial_sum, int v_total) {
  const int n = blockIdx.x;
  const int split = blockIdx.y;
  const int splits = gridDim.y;
  const long row = (long)n * v_total;
  const int per = (v_total + splits - 1) / splits;
  const int v0 = split * per;
  const int v1 = min(v_total, v0 + per);

  float m = -INFINITY;
  float s = 0.0f;
  const int vec0 = v0 + ((4 - (v0 & 3)) & 3);  // first float4-aligned index
  const int vec1 = v1 - ((v1 - vec0) & 3);
  for (int i = v0 + threadIdx.x; i < min(v1, vec0); i += kStatsThreads) {
    const float x = logits[row + i];
    const float mn = fmaxf(m, x);
    s = s * __expf(m - mn) + __expf(x - mn);
    m = mn;
  }
  const float4* lv = reinterpret_cast<const float4*>(logits + row + vec0);
  const int nvec = (vec1 - vec0) / 4;
  for (int i = threadIdx.x; i < nvec; i += kStatsThreads) {
    const float4 x = lv[i];
    float lm = fmaxf(fmaxf(x.x, x.y), fmaxf(x.z, x.w));
    const float mn = fmaxf(m, lm);
    s = s * __expf(m - mn) + __expf(x.x - mn) + __expf(x.y - mn) +
        __expf(x.z - mn) + __expf(x.w - mn);
    m = mn;
  }
  for (int i = vec1 + threadIdx.x; i < v1; i += kStatsThreads) {
    const float x = logits[row + i];
    const float mn = fmaxf(m, x);
    s = s * __expf(m - mn) + __expf(x - mn);
    m = mn;
  }

  // warp then block reduction of (m, s)
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const float om = __shfl_xor_sync(0xffffffffu, m, off);
    const float os = __shfl_xor_sync(0xffffffffu, s, off);
    const float mn = fmaxf(m, om);
    s = s * __expf(m - mn) + os * __expf(om - mn);
    m = mn;
  }
  __shared__ float sm[kStatsThreads / 32], ss[kStatsThreads / 32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) {
    sm[warp] = m;
    ss[warp] = s;
  }
  __syncthreads();
  if (warp == 0) {
    constexpr int kW = kStatsThreads / 32;
    m = lane < kW ? sm[lane] : -INFINITY;
    s = lane < kW ? ss[lane] : 0.0f;
#pragma unroll
    for (int off = kW / 2; off > 0; off >>= 1) {
      const float om = __shfl_xor_sync(0xffffffffu, m, off);
      const float os = __shfl_xor_sync(0xffffffffu, s, off);
      const float mn = fmaxf(m, om);
      s = s * __expf(m - mn) + os * __expf(om - mn);
      m = mn;
    }
    if (lane == 0) {
      partial_max[(long)n * splits + split] = m;
      partial_sum[(long)n * splits + split] = s;
    }
  }
}

}  // namespace

void vocab_ce_stats_launch(const float* logits, float* partial_max,
                           float* partial_sum, int rows, int v,
                           cudaStream_t stream) {
  dim3 grid(rows, kStatsSplits), block(kStatsThreads);
  vocab_ce_stats_kernel<<<grid, block, 0, stream>>>(logits, partial_max,
                                                    partial_sum, v);
}

}  // namespace vocab_ce_train
}  // namespace flashrt_hub
