# fp4-gemm

FlashRT native Blackwell NVFP4 A4W4 GEMM kernels plus a small-M W4A16 tier.

This package consumes packed FP4 E2M1 tensors plus CUTLASS Sm1xx SFA/SFB scale
buffers and produces BF16 output. It is designed to pair with
`flashrt/fp4-fused-ops` and other static low-bit transformer/diffuser runtime
paths.

## Available Functions

- `sfa_size_bytes(rows, dim)`
- `capabilities()`
- `quantize_fp4_sfa_fp16(x, packed=None, sfa=None, is_sfb=False)`
- `quantize_fp4_sfa_bf16(x, packed=None, sfa=None, is_sfb=False)`
- `quantize_fp4_sfa_bf16_pdl(x, packed=None, sfa=None, is_sfb=False)`
- `quantize_fp4_sfa_mse_fp16(x, packed=None, sfa=None, is_sfb=False)`
- `quantize_fp4_sfa_mse_bf16(x, packed=None, sfa=None, is_sfb=False)`
- `quantize_fp4_sfa_padded_bf16(x, alignment=32, is_sfb=False)`
- `pack_nvfp4_weight_bf16(weight, bias=None, alignment=32, mse=False)`
- `quantize_e0m3_sfa_fp16(x, packed=None, sfa=None, is_sfb=False)`
- `dequantize_fp4_sfa_fp16(packed, sfa, out=None, is_sfb=False)`
- `nvfp4_gemm_bf16(a_packed, b_packed, sfa, sfb, alpha=1.0, out=None, variant=-1)`
- `nvfp4_gemm_variant_bf16(a_packed, b_packed, sfa, sfb, alpha=1.0, variant=7, out=None)`
- `nvfp4_gemm_nvfp4(a_packed, b_packed, sfa, sfb, out_packed=None, out_sfa=None)`
- `nvfp4_gemm_fp16(a_packed, b_packed, sfa, sfb, alpha=1.0, out=None, variant=-1)`
- `nvfp4_gemm_geglu_nvfp4_fp16(a_packed, b_interleaved_packed, sfa, sfb, skinny=False, ...)`
- `cutlass_fp4_gemm_geglu_il_hw_v10(...)`
- `nvfp4_gemm_bias_gelu_nvfp4_fp16(a_packed, b_packed, sfa, sfb, bias, ...)`
- `nvfp4_gemm_bias_residual_fp16(a_packed, b_packed, sfa, sfb, bias, residual, out=None)`
- `nvfp4_gemm_bias_bf16(a_packed, b_packed, sfa, sfb, bias, out=None)`
- `nvfp4_gemm_bias_residual_bf16(a_packed, b_packed, sfa, sfb, bias, residual, out=None)`
- `nvfp4_gemm_residual_bf16(a_packed, b_packed, sfa, sfb, residual, alpha=1.0, out=None)`
- `nvfp4_gemm_bias_gelu_bf16(a_packed, b_packed, sfa, sfb, bias, alpha=1.0, out=None)`
- `nvfp4_gemm_bias_gelu_nvfp4(a_packed, b_packed, sfa, sfb, bias, alpha=1.0, out_packed=None, out_sfa=None)`
- `nvfp4_gemm_streamk_bf16(a_packed, b_packed, sfa, sfb, alpha=1.0, out=None)`
- `nvfp4_gemm_streamk_bias_bf16(a_packed, b_packed, sfa, sfb, bias, alpha=1.0, out=None)`
- `fp4_w4a16_linear_bf16(...)` is retained as a compatibility alias
- `adopt_nvfp4_w4a16_marlin(weight_packed, weight_scale, weight_scale_2)`
- `allocate_w4a16_marlin_workspace(device)`
- `nvfp4_w4a16_marlin_bf16(x, weight_marlin, weight_scale_marlin, weight_global_scale, workspace=..., out=None)`
- `fp4_w4a4_gemm_warpsplit_mrows_bf16(a_packed, b_packed, sfa, sfb, ...)`
- `fp4_w4a4_gemm_warpsplit_mrows_pdl_bf16(a_packed, b_packed, sfa, sfb, ...)`
- `e0m3_weight_gemm_fp16(a_packed, b_packed, sfa, sfb, alpha=1.0, a_format=1, out=None)`
- `nvfp4_gemm_relu2_nvfp4(a_packed, b_packed, sfa, sfb, out_packed=None, out_sfa=None)`

## Tensor Contract

- `a_packed`: `torch.uint8`, shape `(M, K / 2)`.
- `b_packed`: `torch.uint8`, shape `(N, K / 2)`.
- `sfa`: `torch.uint8`, CUTLASS SFA layout for `(M, K)`.
- `sfb`: `torch.uint8`, CUTLASS SFB layout for `(N, K)`.
- output: `torch.bfloat16`, shape `(M, N)`.
- Native SM110 PI0.5 epilogues use FP16 outputs/producers; their dtype is
  explicit in the function name.
- `K` must be divisible by 16.
- Targets: Blackwell `sm_110a` (Jetson AGX Thor, CUDA 13+) and `sm_120a`
  (RTX Blackwell, CUDA 12.8+).

`capabilities()` is the artifact-owned source of truth for layouts, scale
factor tiling and alignment. In particular, its scale buffer formula is
`ceil(rows/128) * ceil((cols/16)/4) * 128 * 64` bytes. The public Tensor API
accepts arbitrary positive `M`; individual raw CUTLASS tiles may internally
use an `M=128` tile and must not be treated as the public shape contract.
Unsupported shapes and CUTLASS `can_implement`/initialize/run failures raise a
PyTorch exception. They never print and continue with undefined output.

`variant` selects the CUTLASS schedule:

- `-1`: architecture-aware auto-dispatch (public default).
- `0`: default `<128,128,256>` cooperative schedule.
- `1`: widen `<128,256,128>` schedule, intended for very large `N`.
- `2`: pingpong schedule for A/B testing shape-specific wins.

The canonical linear API, fused `nvfp4_gemm_bias_bf16`, and FP4/SFA
quantize/dequantize helpers are available on both SM110 and SM120. SM110
additionally provides the GROOT N1.7 production epilogues
`nvfp4_gemm_bias_residual_bf16` and
`nvfp4_gemm_bias_gelu_nvfp4`. The latter emits packed FP4 plus CUTLASS SFA so
the following projection can consume it without a BF16 materialization and a
standalone quantization launch. Stream-K and the older BF16 GELU epilogue keep
their existing SM120 dispatch and reject unsupported architectures explicitly.

The SM110 release gate includes the production `(M,N,K)` shapes
`(41,4608,1536)`, `(41,6144,1536)`, and `(41,1536,6144)`, plus the legacy
`M=51` compatibility row. The kernels are the native sources used by FlashRT's
GROOT N1.7 Thor NVFP4 pipeline.

PI0.5 Thor coverage additionally includes FP16 linear projections, compact
GeGLU-to-NVFP4, and SigLIP bias+GELU / bias+residual epilogues. SigLIP's
logical hidden width `4304` is physically zero-padded to `4320` for the FP4
TMA contract; direct `4304` GEMM calls are unsupported. Use
`pack_nvfp4_weight_bf16` for static weights/bias and
`quantize_fp4_sfa_padded_bf16` for a bind-time activation buffer. Both helpers
return logical-shape metadata while producing the physical aligned tensors.
They allocate by design and must not be called from the captured hot path.
E0M3 weight GEMM and the ReLU-squared FP4-output epilogue are SM110-only.
`a_format=1` consumes E2M1 activations; `a_format=0` consumes E0M3 activations.

The additive PI0.5 SM110 BF16 schedule API exposes the production tile IDs
directly:

- `variant=10`: `128x64x256`, tuned for decoder `M=10..64` and narrow `N`;
- `variant=7`: `128x128x256`, tuned for encoder/SigLIP `M=512..970`;
- other IDs are rejected by `nvfp4_gemm_variant_bf16` instead of silently
  selecting an unqualified tile.

`nvfp4_gemm_nvfp4` keeps the GEMM result in packed E2M1 plus SFA form for a
following low-bit combiner/GEMM. Call it once eagerly before CUDA Graph capture
to initialize its immutable scale constant; subsequent launches allocate no
workspace and are graph safe. `nvfp4_gemm_geglu_nvfp4_fp16(..., skinny=True)`
is the public Tensor API for the native `cutlass_fp4_gemm_geglu_il_hw_v10`
interleaved-weight epilogue.

## Minimal Usage

```python
from kernels import get_kernel
import torch

ops = get_kernel("flashrt/fp4-gemm", version=1, trust_remote_code=True)

x = torch.randn((32, 256), device="cuda", dtype=torch.float16)
w = torch.randn((512, 256), device="cuda", dtype=torch.float16)

a_packed, sfa = ops.quantize_fp4_sfa_fp16(x, is_sfb=False)
b_packed, sfb = ops.quantize_fp4_sfa_fp16(w, is_sfb=True)

y = ops.nvfp4_gemm_bf16(a_packed, b_packed, sfa, sfb, alpha=1.0)

# PI0.5 Thor decoder schedule, BF16 output.
y_decoder = ops.nvfp4_gemm_variant_bf16(
    a_packed, b_packed, sfa, sfb, variant=10
)

# Split-GU path: preserve packed FP4 output for the next fused consumer.
out_packed, out_sfa = ops.nvfp4_gemm_nvfp4(
    a_packed, b_packed, sfa, sfb
)
```

For BF16 model activations, use the direct producer so the hot path does not
materialize an intermediate FP16 tensor:

```python
x_bf16 = torch.randn((1, 5120), device="cuda", dtype=torch.bfloat16)
a_packed, sfa = ops.quantize_fp4_sfa_bf16(x_bf16)

# Logical SigLIP FC1 weight (4304, 1152) is packed as physical (4320, 1152).
w = torch.randn((4304, 1152), device="cuda", dtype=torch.bfloat16)
bias = torch.randn((4304,), device="cuda", dtype=torch.bfloat16)
w_packed, w_sfb, bias_padded, logical_shape = \
    ops.pack_nvfp4_weight_bf16(w, bias)
assert logical_shape == (4304, 1152)
assert w_packed.shape[0] == bias_padded.shape[0] == 4320
```

The BF16 entry writes the same E2M1 bytes and CUTLASS SFA/SFB layout as
`quantize_fp4_sfa_fp16(x_bf16.to(torch.float16))` for finite FP16-range
inputs. It is an additive API; the existing FP16 producer remains unchanged.
On SM110 this entry dispatches the vectorized 16-element-block implementation
(two 16-byte BF16 loads and one 8-byte packed store), not the scalar fallback.

The FP16 and BF16 `quantize_fp4_sfa_mse_*` functions are intended for
offline/bind-time weight packing. They search a compact per-block scale set and
are release-gated to produce no higher reconstruction MSE than the matching
default RTN packer. They are not runtime activation hot-path operations.

The quantize/dequantize helpers are included for examples and validation. A
production runtime should keep weights prepacked and should avoid quantizing in
the hot path unless that producer kernel is part of the intended low-bit block.

Use the bias/GELU and residual variants to avoid returning to BF16
elementwise code between low-bit GEMMs. Stream-K variants are selected only
for the validated large down-projection shapes; unsupported shapes reject
rather than silently selecting a losing schedule.

For M=1 decode, call `fp4_repack_b_interleaved_sm120` once when binding a
static packed weight, then use `fp4_w4a4_gemv_warpsplit_interleaved_bf16` in
the hot path. The default 8-warp/3-stage schedule is the qualified
out-of-cache profile; explicit `warps` and `stages` remain available for
shape-specific tuning.

`nvfp4_gemm_m256_bf16` is an explicit SM120 large-M tier. Query
`nvfp4_gemm_m256_workspace_size` and allocate the workspace before graph
capture. Runtime dispatchers must read `capabilities()`; M>=512 alone is not
a performance qualification for every N/K pair. The current persistent M256
schedule reports exactly zero workspace bytes. Its fake implementation and
compile path preserve that contract under FakeTensor and
`torch.compile(fullgraph=True)`.

`fp4_w4a4_gemm_warpsplit_mrows_bf16` is the SM120 speculative-verify tier.
Its `_pdl_` twin and `quantize_fp4_sfa_bf16_pdl` are additive programmatic
dependent launch variants. They are bit-identical to the non-PDL entries and
are intended for static decode chains; chain-level E2E timing, not standalone
latency, decides whether a runtime enables them.
It serves `M=1..16` with the standard packed weight and scale layout, so it
does not require the duplicate interleaved weight used by the M=1 decode
tier. Read the exact alignment and row range from `capabilities()`.

The W4A16 tier keeps logits and other accuracy-sensitive activations in BF16
while reading static ModelOpt NVFP4 weights. Convert each weight once at model
bind time; the hot operation takes caller-owned output and lock workspace and
performs no allocation:

```python
weight_marlin, scale_marlin, global_marlin, workspace = \
    ops.adopt_nvfp4_w4a16_marlin(
        weight_packed,       # uint8 [N, K/2], E2M1
        weight_scale,        # float8_e4m3fn [N, K/16]
        weight_scale_2,      # float32 scalar
    )

out = torch.empty((8, weight_packed.shape[0]), device="cuda",
                  dtype=torch.bfloat16)
y = ops.nvfp4_w4a16_marlin_bf16(
    x_bf16,
    weight_marlin,
    scale_marlin,
    global_marlin,
    workspace=workspace,
    out=out,
)
```

This SM120 tier supports `1<=M<=16`, `K%128==0`, and `N%64==0`. Adoption is
outside the hot path. The runtime call is CUDA Graph safe and has a fake
implementation for `torch.compile(fullgraph=True)`.

## Validation

```bash
python fp4-gemm/tests/test_fp4_gemm.py --backend source --mode full
python fp4-gemm/tests/test_fp4_gemm.py --backend installed --mode full \
  --artifact fp4-gemm/build/torch211-cxx11-cu128-x86_64-linux
python fp4-gemm/benchmarks/benchmark.py --backend installed --mode headline \
  --artifact fp4-gemm/build/torch211-cxx11-cu128-x86_64-linux

# Accuracy-sensitive small-M full-vocabulary W4A16 gate
python fp4-gemm/benchmarks/benchmark.py --backend installed \
  --mode w4a16-vocab \
  --artifact fp4-gemm/build/torch213-cxx11-cu130-x86_64-linux

# Thor model-shape gate
python fp4-gemm/tests/test_fp4_gemm.py --backend installed \
  --mode thor-models \
  --artifact fp4-gemm/build/torch211-cxx11-cu130-aarch64-linux
```

The correctness reference dequantizes the same FP4/SFA and FP4/SFB inputs used
by the kernel, then computes the PyTorch GEMM reference from those dequantized
low-bit values.

The W4A16 implementation is adapted from the Apache-2.0 Marlin backend in
[vLLM](https://github.com/vllm-project/vllm), itself derived from
[Marlin](https://github.com/IST-DASLab/marlin). FlashRT exposes a standalone
Tensor API and does not require vLLM at runtime. When vLLM is installed, the
benchmark additionally requires bit-exact output against its native Marlin
operator on the same adopted tensors.

The producer gate also checks the BF16 direct entry byte-for-byte against the
established FP16 compatibility chain at decode widths 5120, 6144 and 17408,
plus multi-row activation and SFB layouts.

The SM110 PI0.5 gate additionally covers all decoder v10, encoder/SigLIP v7,
FP4-output, MSE-packing and CUDA Graph replay paths. Performance is compared
against the matching FlashRT native v7/v10/FP4-output kernels on the same Thor,
not only against PyTorch eager.

Release artifacts cover the standard x86 matrix (Torch 2.11/2.12/2.13 over
the supported CUDA 12.8/13.x variants) and native Thor aarch64 builds for
Torch 2.11 and 2.13 with CUDA 13.0. Consumers should request `version=1`.
