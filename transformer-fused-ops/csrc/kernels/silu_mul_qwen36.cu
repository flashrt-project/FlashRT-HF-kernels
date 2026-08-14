// SiLU-gate elementwise multiply for Qwen3.6 SwiGLU MLP.
//
// Computes out[i] = silu(gate[i]) * up[i] elementwise, bf16 in/out,
// fp32 internal. Replaces the F.silu(gate) * up Python composite that
// allocates two intermediates per call (silu output + multiply output)
// in the hot path.

#include "silu_mul_qwen36.cuh"

namespace flash_rt::kernels {

namespace {

constexpr int kThreadsX = 256;

__device__ __forceinline__ float silu_f32(float x) {
  // Use expf (full-precision intrinsic), not __expf (fast/low-precision)
  // -- matches PyTorch's F.silu precision so cos vs HF stays >= 0.998.
  return x / (1.0f + expf(-x));
}

__global__ void silu_mul_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ out,
    int n)
{
  const int idx = blockIdx.x * kThreadsX + threadIdx.x;
  if (idx >= n) return;
  const float g = __bfloat162float(gate[idx]);
  const float u = __bfloat162float(up[idx]);
  // Match PyTorch's two-step rounding pattern: silu(g) returns bf16
  // (one rounding), then bf16 * bf16 multiply rounds again. The
  // model was trained with this exact pattern, so a more-precise
  // single-rounding fused version subtly drifts cos. Round-trip
  // through bf16 between silu and multiply to reproduce it.
  const float silu_g = silu_f32(g);
  const float silu_g_bf_rt = __bfloat162float(__float2bfloat16(silu_g));
  out[idx] = __float2bfloat16(silu_g_bf_rt * u);
}

__global__ void sigmoid_mul_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ out,
    int n)
{
  const int idx = blockIdx.x * kThreadsX + threadIdx.x;
  if (idx >= n) return;
  const float g = __bfloat162float(gate[idx]);
  const float xv = __bfloat162float(x[idx]);
  const float sig = 1.0f / (1.0f + expf(-g));
  const float sig_bf_rt = __bfloat162float(__float2bfloat16(sig));
  out[idx] = __float2bfloat16(xv * sig_bf_rt);
}

__global__ void per_head_sigmoid_gate_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ out,
    int heads,
    int head_dim2,
    int pairs) {
  const int pair = blockIdx.x * kThreadsX + threadIdx.x;
  if (pair >= pairs) return;
  const int head_linear = pair / head_dim2;
  const int h = head_linear % heads;
  const float g = __bfloat162float(gate[head_linear]);
  const __nv_bfloat16 factor =
      __float2bfloat16(2.0f / (1.0f + expf(-g)));
  const __nv_bfloat162 xv = reinterpret_cast<const __nv_bfloat162*>(x)[pair];
  const __nv_bfloat162 fv = __halves2bfloat162(factor, factor);
  reinterpret_cast<__nv_bfloat162*>(out)[pair] = __hmul2(xv, fv);
}

}  // namespace

void silu_mul_qwen36_bf16(
    const __nv_bfloat16* gate,
    const __nv_bfloat16* up,
    __nv_bfloat16* out,
    int n,
    cudaStream_t stream)
{
  const int grid = (n + kThreadsX - 1) / kThreadsX;
  silu_mul_kernel<<<grid, kThreadsX, 0, stream>>>(gate, up, out, n);
}

void sigmoid_mul_qwen36_bf16(
    const __nv_bfloat16* gate,
    const __nv_bfloat16* x,
    __nv_bfloat16* out,
    int n,
    cudaStream_t stream)
{
  const int grid = (n + kThreadsX - 1) / kThreadsX;
  sigmoid_mul_kernel<<<grid, kThreadsX, 0, stream>>>(gate, x, out, n);
}

void per_head_sigmoid_gate_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* gate,
    __nv_bfloat16* out,
    int rows,
    int heads,
    int head_dim,
    cudaStream_t stream) {
  const int pairs = rows * heads * (head_dim / 2);
  const int grid = (pairs + kThreadsX - 1) / kThreadsX;
  per_head_sigmoid_gate_kernel<<<grid, kThreadsX, 0, stream>>>(
      x, gate, out, heads, head_dim / 2, pairs);
}

}  // namespace flash_rt::kernels
