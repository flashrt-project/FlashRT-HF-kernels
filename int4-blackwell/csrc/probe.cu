// INT4/E0M3 unlock probe for SM120 (RTX 5090).
//
// Two kernels, both built around the same instruction our v19sf kernels use:
//   mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64
//   .row.col.f32.e2m1.e2m1.f32.ue4m3
// which ptxas lowers to SASS:
//   OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X
//
// The claim under test: bits 78/79 of the 128-bit SASS encoding select the
// A/B operand element format (0 = E2M1, 1 = E0M3 aka INT4, codebook -7..7).
// We compile this file to a cubin, flip those bits with patch_cubin.py, and
// run both cubins through the same driver-API runner.
//
//   codebook_probe : fills A with one nibble value broadcast everywhere,
//                    B with another, SF = 1.0 (0x38 ue4m3). Every element of
//                    the 16x8 output tile must equal 64*dec(a)*dec(b), so
//                    D[0]/64 recovers the decode table without any fragment
//                    layout reasoning.
//   perf_mma       : register-resident back-to-back MMA throughput loop,
//                    4 independent accumulator chains per warp.

#include <cstdint>

__device__ __forceinline__
void mma_e2m1_4x(float &d0, float &d1, float &d2, float &d3,
                 uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                 uint32_t b0, uint32_t b1,
                 uint32_t sfa, uint32_t sfb)
{
  constexpr uint16_t bidA = 0, tidA = 0, bidB = 0, tidB = 0;
  asm volatile(
    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64"
    ".row.col.f32.e2m1.e2m1.f32.ue4m3 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
    "{%14},{%15,%16},{%17},{%18,%19};\n"
    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
      "f"(d0), "f"(d1), "f"(d2), "f"(d3),
      "r"(sfa), "h"(bidA), "h"(tidA),
      "r"(sfb), "h"(bidB), "h"(tidB));
}

extern "C" __global__
void codebook_probe(uint32_t aword, uint32_t bword,
                    uint32_t sfa, uint32_t sfb, float *D)
{
  float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
  mma_e2m1_4x(d0, d1, d2, d3,
              aword, aword, aword, aword, bword, bword, sfa, sfb);
  int lane = threadIdx.x & 31;
  D[lane * 4 + 0] = d0;
  D[lane * 4 + 1] = d1;
  D[lane * 4 + 2] = d2;
  D[lane * 4 + 3] = d3;
}

extern "C" __global__
void perf_mma(int iters, uint32_t aword, uint32_t bword,
              uint32_t sfa, uint32_t sfb, float *out)
{
  uint32_t a0 = aword, a1 = aword ^ 0x11111111u,
           a2 = aword ^ 0x22222222u, a3 = aword ^ 0x33333333u;
  uint32_t b0 = bword, b1 = bword ^ 0x11111111u;
  float acc[4][4];
#pragma unroll
  for (int g = 0; g < 4; ++g)
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[g][i] = 0.f;

  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int g = 0; g < 4; ++g)
      mma_e2m1_4x(acc[g][0], acc[g][1], acc[g][2], acc[g][3],
                  a0, a1, a2, a3, b0, b1, sfa, sfb);
  }

  float s = 0.f;
#pragma unroll
  for (int g = 0; g < 4; ++g)
#pragma unroll
    for (int i = 0; i < 4; ++i) s += acc[g][i];
  if (s == 12345.678f) out[threadIdx.x] = s;  // never taken; defeats DCE
}
