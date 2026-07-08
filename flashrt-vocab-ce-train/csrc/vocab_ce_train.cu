// SPDX-License-Identifier: Apache-2.0
//
// Streaming linear-CE forward for small N over a huge fp32 vocabulary head.
//
// The (V, H) weight is the traffic; it is read exactly once. Each CTA owns
// kVTile vocab rows and keeps ALL N hidden rows resident (staged per
// H-chunk in shared memory), so the weight is never re-read per hidden-row
// tile. Staging is double-buffered with cp.async so the global-memory
// latency of the next chunk hides under the current chunk's math. The
// epilogue writes the logits tile (needed by the backward), gathers the
// label logits, and emits one online-softmax partial (max, sumexp) per
// (row, vocab-tile) for a cheap downstream merge.

#include <cuda_pipeline_primitives.h>

#include "vocab_ce_train.cuh"

namespace flashrt_hub {
namespace vocab_ce_train {

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kPad = 4;  // 16B row pad: float4-aligned and bank-staggered

template <int kHChunk>
__device__ __forceinline__ void stage_chunk(const float* __restrict__ hidden,
                                            const float* __restrict__ weight,
                                            float* __restrict__ h_buf,
                                            float* __restrict__ w_buf, int v0,
                                            int h0, int rows, int h) {
  constexpr int kStride = kHChunk + kPad;
  // hidden rows: rows x kHChunk floats, 16B copies
  for (int idx = threadIdx.x; idx < rows * (kHChunk / 4); idx += kThreads) {
    const int r = idx / (kHChunk / 4);
    const int c4 = idx - r * (kHChunk / 4);
    __pipeline_memcpy_async(h_buf + r * kStride + c4 * 4,
                            hidden + (long)r * h + h0 + c4 * 4, 16);
  }
  // weight tile: kVTile x kHChunk floats
  for (int idx = threadIdx.x; idx < kVTile * (kHChunk / 4); idx += kThreads) {
    const int r = idx / (kHChunk / 4);
    const int c4 = idx - r * (kHChunk / 4);
    __pipeline_memcpy_async(w_buf + r * kStride + c4 * 4,
                            weight + (long)(v0 + r) * h + h0 + c4 * 4, 16);
  }
}

template <int kJn, int kHChunk, int kStages>
__global__ void __launch_bounds__(kThreads) vocab_ce_fwd_kernel(
    const float* __restrict__ hidden, const float* __restrict__ weight,
    const long* __restrict__ labels, float* __restrict__ logits,
    float* __restrict__ partial_max, float* __restrict__ partial_sum,
    float* __restrict__ label_logit, int rows, int v_total, int h) {
  extern __shared__ float smem[];
  constexpr int kStride = kHChunk + kPad;
  const int h_buf_elems = rows * kStride;
  const int w_buf_elems = kVTile * kStride;
  // layout: h[0..kStages-1], w[0..kStages-1]
  float* h_bufs[kStages];
  float* w_bufs[kStages];
#pragma unroll
  for (int i = 0; i < kStages; ++i) {
    h_bufs[i] = smem + i * h_buf_elems;
    w_bufs[i] = smem + kStages * h_buf_elems + i * w_buf_elems;
  }

  const int v0 = blockIdx.x * kVTile;
  const int lane = threadIdx.x & 31;  // vocab row within the tile
  const int grp = threadIdx.x >> 5;   // hidden-row group (stride kWarps)
  const int num_tiles = gridDim.x;
  const int num_chunks = h / kHChunk;

  float acc[kJn];
  const int jn = (rows - grp + kWarps - 1) / kWarps;  // rows for this group
#pragma unroll
  for (int j = 0; j < kJn; ++j) acc[j] = 0.0f;

  // prologue: fill kStages-1 buffers ahead
  for (int k = 0; k < kStages - 1 && k < num_chunks; ++k) {
    stage_chunk<kHChunk>(hidden, weight, h_bufs[k], w_bufs[k], v0,
                         k * kHChunk, rows, h);
    __pipeline_commit();
  }

  for (int k = 0; k < num_chunks; ++k) {
    const int buf = k % kStages;
    const int ahead = k + kStages - 1;
    if (ahead < num_chunks) {
      stage_chunk<kHChunk>(hidden, weight, h_bufs[ahead % kStages],
                           w_bufs[ahead % kStages], v0, ahead * kHChunk, rows,
                           h);
      __pipeline_commit();
      __pipeline_wait_prior(kStages - 1);
    } else {
      __pipeline_wait_prior(num_chunks - 1 - k);
    }
    __syncthreads();

    const float4* wrow =
        reinterpret_cast<const float4*>(w_bufs[buf] + lane * kStride);
    const float4* hrow =
        reinterpret_cast<const float4*>(h_bufs[buf] + grp * kStride);
    constexpr int kRowStride4 = kWarps * kStride / 4;
#pragma unroll 2
    for (int hh = 0; hh < kHChunk / 4; ++hh) {
      const float4 wv = wrow[hh];
#pragma unroll
      for (int j = 0; j < kJn; ++j) {
        if (j < jn) {
          const float4 hv = hrow[j * kRowStride4 + hh];
          acc[j] += wv.x * hv.x + wv.y * hv.y + wv.z * hv.z + wv.w * hv.w;
        }
      }
    }
    __syncthreads();  // buf is re-staged kStages iterations from now
  }

  // epilogue: logits write, label gather, per-tile online-softmax partials
  for (int j = 0; j < jn && j < kJn; ++j) {
    const int n = grp + j * kWarps;
    const float logit = acc[j];
    logits[(long)n * v_total + v0 + lane] = logit;
    if (labels[n] == (long)(v0 + lane)) label_logit[n] = logit;

    float m = logit;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    float e = __expf(logit - m);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      e += __shfl_xor_sync(0xffffffffu, e, off);
    if (lane == 0) {
      partial_max[(long)n * num_tiles + blockIdx.x] = m;
      partial_sum[(long)n * num_tiles + blockIdx.x] = e;
    }
  }
}

template <int kHChunk, int kStages>
int smem_bytes_for(int rows) {
  return kStages * (rows + kVTile) * (kHChunk + kPad) * (int)sizeof(float);
}

template <int kHChunk, int kStages>
void launch_all_jn(const float* hidden, const float* weight,
                   const long* labels, float* logits, float* partial_max,
                   float* partial_sum, float* label_logit, int rows, int v,
                   int h, cudaStream_t stream) {
  const int smem = smem_bytes_for<kHChunk, kStages>(rows);
  const int jn = (rows + kWarps - 1) / kWarps;
  dim3 grid(v / kVTile), block(kThreads);
#define CASE(J)                                                               \
  case J: {                                                                   \
    static bool configured_##J = false;                                       \
    if (!configured_##J) {                                                    \
      cudaFuncSetAttribute(vocab_ce_fwd_kernel<J, kHChunk, kStages>,          \
                           cudaFuncAttributeMaxDynamicSharedMemorySize,       \
                           smem_bytes_for<kHChunk, kStages>(kMaxRows));       \
      configured_##J = true;                                                  \
    }                                                                         \
    vocab_ce_fwd_kernel<J, kHChunk, kStages><<<grid, block, smem, stream>>>(  \
        hidden, weight, labels, logits, partial_max, partial_sum,             \
        label_logit, rows, v, h);                                             \
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
  // Adaptive chunking: keep two CTAs per SM resident where possible; the
  // cp.async double buffer hides the global-memory latency. Deeper
  // pipelines with smaller chunks measured slower (sync + commit overhead
  // outweighs the extra overlap at these tile sizes).
  if (rows <= 64 && h % 64 == 0) {
    launch_all_jn<64, 2>(hidden, weight, labels, logits, partial_max,
                         partial_sum, label_logit, rows, v, h, stream);
  } else {
    launch_all_jn<32, 2>(hidden, weight, labels, logits, partial_max,
                         partial_sum, label_logit, rows, v, h, stream);
  }
}

}  // namespace vocab_ce_train
}  // namespace flashrt_hub
