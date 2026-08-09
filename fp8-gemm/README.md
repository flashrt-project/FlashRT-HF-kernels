# fp8-gemm

FlashRT native CUDA FP8 GEMV/GEMM kernels for low-latency transformer and
diffuser linear layers on NVIDIA Ada SM89 and Blackwell SM110/SM120 GPUs.

This package exposes the hand-tuned FP8 E4M3 decode and small-M kernels as
Tensor APIs for Hugging Face Kernel Hub. It is intended for model runtimes that
already hold activations and weights in FP8 and want a low-overhead BF16 output
linear path.

## Available Functions

- `fp8_linear_bf16(input, weight, alpha=1.0, out=None, variant=0)`
- `fp8_linear_residual_bf16(input, weight, residual, alpha=1.0, variant=0)`
- `fp8_linear_bias_bf16(input, weight, bias, alpha=1.0, out=None)`
- `fp8_linear_bias_residual_bf16(input, weight, bias, residual, alpha=1.0)`
- `fp8_linear_bias_gelu_bf16(input, weight, bias, alpha=1.0, out=None)`
- `fp8_blockwise_linear_bf16(input, weight, input_scale, weight_scale, out=None)`
- `fp8_blockwise_swiglu_quantize_fp8(input, gate_up_weight, input_scale, gate_up_weight_scale, output=None, output_scale=None)`
- `select_fp8_linear_tile(m, n, k, variant=0)`

Tensor contract:

- `input`: `torch.float8_e4m3fn`, shape `(M, K)`, contiguous CUDA tensor.
- `weight`: `torch.float8_e4m3fn`, shape `(N, K)`, contiguous CUDA tensor.
- `out`: `torch.bfloat16`, shape `(M, N)`.
- `residual`: `torch.bfloat16`, shape `(1, N)` or `(N,)`, only supported for
  the `M=1` decode GEMV path.
- `K % 16 == 0`; SM120 additionally requires `K % 32 == 0`.
- On SM120, `M == 1` uses dedicated GEMV and `2 <= M <= 64` uses small-M
  GEMM tiles.
- On SM110 (Jetson AGX Thor), the per-tensor API uses the production FlashRT
  CUTLASS Sq/T1/Wide family and supports the validated model-shape matrix from
  decode through large vision/backbone rows. The large-M production band is
  validated from `M=65` through `M=1024`, including PI0.5 prefill QKV, O,
  gate/up, and down projections at `M=712..970`. `N` and `K` must be divisible
  by 16.
- The three BF16 bias APIs are SM110-only. They accept BF16 `(N,)` bias and
  preserve the same row-major FP8 `(M,K)` input and `(N,K)` weight contract.
  The residual API updates a BF16 `(M,N)` tensor in place. The GELU API uses
  the tanh approximation.
- SM110 `variant=0` is the production auto dispatcher. Diagnostic variants are
  `1=Sq`, `2=T1`, and `3=Wide`; they are correctness-tested but should not be
  pinned by model integrations without a shape-specific benchmark.
- The per-tensor kernels use Blackwell FP8 MMA instructions and are not valid
  for SM89. SM89 support is provided by the blockwise API below.
- `alpha` is a host float. For per-tensor FP8 quantization, pass
  `float(input_scale * weight_scale)` from your static calibration metadata.

The blockwise API uses a separate contract:

- `input`: FP8 E4M3 `(M, K)`.
- `weight`: FP8 E4M3 `(N, K)`.
- `input_scale`: FP32 `(M, K / 128)`.
- `weight_scale`: FP32 `(N / 128, K / 128)`.
- `N` and `K` must be divisible by 128; `M` is unrestricted.
- Output is BF16 `(M, N)`.
- On SM89, the blockwise API dispatches to the production FlashRT native
  `mma.sync.aligned.m16n8k32` GEMM/GEMV implementation.
- On SM120, it dispatches to the production FlashRT CUTLASS block-scaled
  implementation.
- SM110 is intentionally not claimed by the blockwise API; use the per-tensor
  static-scale path there. Other architectures are rejected explicitly.

The fused SM89 producer accepts FP8 `(M,K)` input, FP8 `(2*N,K)` gate/up
weight, block-128 FP32 scales, and returns FP8 `(M,N)` plus FP32 `(M,N/128)`
output scales. Its public range is `1 <= M <= 256` with `N` and `K` divisible
by 128. It is rejected explicitly on non-SM89 GPUs.

## Minimal Usage

```python
from kernels import get_kernel
import torch

ops = get_kernel("flashrt/fp8-gemm", version=1, trust_remote_code=True)

x = torch.randn((16, 4096), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
w = torch.randn((8192, 4096), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)

y = ops.fp8_linear_bf16(x, w, alpha=1.0)
```

SM110 bias epilogues:

```python
bias = torch.randn((8192,), device="cuda", dtype=torch.bfloat16)
residual = torch.randn((16, 8192), device="cuda", dtype=torch.bfloat16)

y = ops.fp8_linear_bias_bf16(x, w, bias, alpha=1.0)
ops.fp8_linear_bias_residual_bf16(x, w, bias, residual, alpha=1.0)
y_gelu = ops.fp8_linear_bias_gelu_bf16(x, w, bias, alpha=1.0)
```

Warm each distinct SM110 bias shape once before CUDA Graph capture. The
cuBLASLt fallback lazily creates and caches its descriptor, algorithm, and
workspace on the first call; replay itself performs no allocation.

Decode residual path:

```python
x = torch.randn((1, 4096), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
w = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
residual = torch.zeros((1, 4096), device="cuda", dtype=torch.bfloat16)

ops.fp8_linear_residual_bf16(x, w, residual, alpha=1.0)
```

Block-128 scaling:

```python
m, k, n = 51, 1536, 1536
x = torch.randn((m, k), device="cuda").to(torch.float8_e4m3fn)
w = torch.randn((n, k), device="cuda").to(torch.float8_e4m3fn)
x_scale = torch.ones((m, k // 128), device="cuda", dtype=torch.float32)
w_scale = torch.ones((n // 128, k // 128), device="cuda", dtype=torch.float32)

y = ops.fp8_blockwise_linear_bf16(x, w, x_scale, w_scale)
```

## Validation

```bash
python fp8-gemm/tests/test_fp8_gemm.py --backend source --mode full
python fp8-gemm/benchmarks/benchmark.py --backend source --mode headline
python fp8-gemm/benchmarks/benchmark.py --backend source --mode pi05-prefill
python fp8-gemm/benchmarks/benchmark_bias.py --backend source
```

The SM110 full sweep covers PI0.5, GROOT N1.6/N1.7, Cosmos Edge, and LingBot
VLA projection families, plus decode, generic small-M, the `M=65` large-M
boundary, and the three SigLIP bias epilogues. Public
benchmark tables are only updated after source correctness, installed artifact
correctness, shape/tile sweeps, `torch.compile(fullgraph=True)`, CUDA Graph
replay, and parity against the original FlashRT native pointer entry pass.
