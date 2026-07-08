// SPDX-License-Identifier: Apache-2.0
//
// Streaming linear-CE forward for small N over a huge fp32 vocabulary head.
//
// The (V, H) weight is the traffic; it is read exactly once. Each CTA owns
// kVTile vocab rows and keeps ALL N hidden rows resident (staged per
// H-chunk in shared memory), so the weight is never re-read per hidden-row
// tile. The epilogue writes the logits tile (needed by the backward),
// gathers the label logits, and emits one online-softmax partial
// (max, sumexp) per (row, vocab-tile) for a cheap downstream merge.

#include "vocab_ce_train.cuh"

namespace flashrt_hub {
namespace vocab_ce_train {

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kHChunk = 128;
constexpr int kPad = 4;  // 16B row pad: float4-aligned and bank-staggered

template <int kJn>
__global__ void __launch_bounds__(kThreads) vocab_ce_fwd_kernel(
    const float* __restrict__ hidden, const float* __restrict__ weight,
    const long* __restrict__ labels, float* __restrict__ logits,
    float* __restrict__ partial_max, float* __restrict__ partial_sum,
    float* __restrict__ label_logit, int rows, int v_total, int h) {
  extern __shared__ float smem[];
  float* h_s = smem;                                    // rows x (kHChunk+1)
  float* w_s = smem + rows * (kHChunk + kPad);          // kVTile x (kHChunk+1)

  const int v0 = blockIdx.x * kVTile;
  const int lane = threadIdx.x & 31;   // vocab row within the tile
  const int grp = threadIdx.x >> 5;    // hidden-row group (stride kWarps)
  const int num_tiles = gridDim.x;

  // per-thread accumulators over this thread's hidden rows; kJn is the
  // exact ceil(rows / kWarps), so no dead unroll slots
  float acc[kJn];
  const int jn = (rows - grp + kWarps - 1) / kWarps;  // rows for this group
#pragma unroll
  for (int j = 0; j < kJn; ++j) acc[j] = 0.0f;

  for (int h0 = 0; h0 < h; h0 += kHChunk) {
    // stage the hidden chunk (rows x kHChunk) and the weight tile
    // (kVTile x kHChunk); both padded by one float per row
    for (int idx = threadIdx.x; idx < rows * kHChunk; idx += kThreads) {
      const int r = idx / kHChunk;
      const int c = idx - r * kHChunk;
      h_s[r * (kHChunk + kPad) + c] = hidden[(long)r * h + h0 + c];
    }
    {
      // one warp per weight row segment: lane-consecutive = coalesced
      const int rows_per_warp = kVTile / kWarps;  // 4
      for (int rr = 0; rr < rows_per_warp; ++rr) {
        const int vrow = grp * rows_per_warp + rr;
        for (int c = lane; c < kHChunk; c += 32) {
          w_s[vrow * (kHChunk + kPad) + c] = weight[(long)(v0 + vrow) * h + h0 + c];
        }
      }
    }
    __syncthreads();

    // float4 smem reads (the +kPad row padding is 4 floats = 16B, so rows
    // stay vector-aligned); exact kJn unroll keeps everything in registers.
    const float4* wrow =
        reinterpret_cast<const float4*>(w_s + lane * (kHChunk + kPad));
    const float4* hrow =
        reinterpret_cast<const float4*>(h_s + grp * (kHChunk + kPad));
    constexpr int kRowStride4 = kWarps * (kHChunk + kPad) / 4;
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
    __syncthreads();
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

}  // namespace

void vocab_ce_fwd_launch(const float* hidden, const float* weight,
                         const long* labels, float* logits, float* partial_max,
                         float* partial_sum, float* label_logit, int rows,
                         int v, int h, cudaStream_t stream) {
  const int smem_bytes =
      (rows * (kHChunk + kPad) + kVTile * (kHChunk + kPad)) * (int)sizeof(float);
  static int configured_bytes = 0;
  if (smem_bytes > configured_bytes) {
#define SETATTR(J)                                                           \
  cudaFuncSetAttribute(vocab_ce_fwd_kernel<J>,                               \
                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes)
    SETATTR(1); SETATTR(2); SETATTR(3); SETATTR(4); SETATTR(5); SETATTR(6);
    SETATTR(7); SETATTR(8); SETATTR(9); SETATTR(10); SETATTR(11); SETATTR(12);
    SETATTR(13); SETATTR(14); SETATTR(15); SETATTR(16);
#undef SETATTR
    configured_bytes = smem_bytes;
  }
  dim3 grid(v / kVTile), block(kThreads);
  const int jn = (rows + kWarps - 1) / kWarps;
#define CASE(J)                                                              \
  case J:                                                                    \
    vocab_ce_fwd_kernel<J><<<grid, block, smem_bytes, stream>>>(             \
        hidden, weight, labels, logits, partial_max, partial_sum,            \
        label_logit, rows, v, h);                                            \
    break
  switch (jn) {
    CASE(1); CASE(2); CASE(3); CASE(4); CASE(5); CASE(6); CASE(7); CASE(8);
    CASE(9); CASE(10); CASE(11); CASE(12); CASE(13); CASE(14); CASE(15);
    default:
      vocab_ce_fwd_kernel<16><<<grid, block, smem_bytes, stream>>>(
          hidden, weight, labels, logits, partial_max, partial_sum,
          label_logit, rows, v, h);
  }
#undef CASE
}

}  // namespace vocab_ce_train
}  // namespace flashrt_hub
