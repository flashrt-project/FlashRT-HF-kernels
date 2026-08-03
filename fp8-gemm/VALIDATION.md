# Validation

Date: June 20, 2026

Local environment:

- GPU: NVIDIA GeForce RTX 5090
- PyTorch: 2.9.1+cu128
- CUDA runtime reported by PyTorch: 12.8
- Source build target: `sm_120a`

## Source Correctness

Command:

```bash
python fp8-gemm/tests/test_fp8_gemm.py --backend source --mode full
```

Result: 14/14 checks passed, plus the blockwise custom op passed
`torch.compile(fullgraph=True)` with bitwise-equal output to the eager wrapper.

Covered public v1 rows:

- M=1 decode GEMV: `K in {512,4096}`, `N in {512,2048,8192}`
- small-M GEMM: `M in {8,16,32,64}` with representative
  transformer/diffuser-adjacent `K,N` rows
- M=1 residual-add GEMV
- block-128 scaled FP8 GEMM at:
  - `(M,K,N)=(1,1024,1024)`
  - `(51,1536,1536)`
  - `(277,2048,2048)`
  - `(1024,1152,1152)`
  - `(2520,3072,3072)`
  - `(128,4096,12288)`

Metrics:

- `max_abs`
- `mean_abs`
- `p99_abs`
- cosine similarity
- output dtype
- tolerance

The blockwise rows use the stricter gate:

- `max_abs <= 0.0625`
- `mean_abs <= 0.003`
- `p99_abs <= 0.015625`
- cosine similarity `>= 0.9999`

The release benchmark also compares the Tensor wrapper against an independent
binding of the original FlashRT pointer API. Matching source code alone is not
treated as proof of zero wrapper overhead.

## Source Benchmark

Command:

```bash
python fp8-gemm/benchmarks/benchmark.py \
  --backend source --mode headline --warmup 20 --iterations 100 --compile-ref
```

Result: all public rows passed. Headline rows are recorded in
`benchmarks/RESULTS.md`.

## Architecture Scope Boundary

On SM120, the public per-tensor path supports `M=1` and `2 <= M <= 64`.
The blockwise path retains its independent unrestricted-M contract.

On SM110, the public per-tensor path uses the production CUTLASS Sq/T1/Wide
family and supports the validated model-shape matrix. The current full sweep
covers `M in {1,8,16,32,51,64,105,277,1024}` and representative `K,N` rows
from PI0.5, GROOT N1.6/N1.7, Cosmos Edge, and LingBot VLA. SM110 blockwise
scaling is not claimed.

## Thor SM110 Increment

Validated August 2, 2026:

- GPU: NVIDIA Thor, compute capability 11.0;
- PyTorch: 2.11.0+cu130;
- CUDA: 13.0;
- CUTLASS: 4.5.2, matching the current `kernel-builder` `cutlass_4_5`
  dependency;
- pinned builder: `e9152aa24e0d99eca255ca9f1beb996de32f9ca4`;
- source correctness: 23/23;
- locally installed aarch64 artifact correctness: 23/23;
- `torch.compile(fullgraph=True)`: exact output parity;
- CUDA Graph capture/replay: exact output parity;
- original SM120 source regression on RTX 5090: 14/14.

The 23 Thor rows include 20 production auto-dispatch checks and three forced
Sq/T1/Wide diagnostics. Ordinary GEMMs were bitwise equal to the FP32
accumulation reference after BF16 output conversion. The residual row passed
with `max_abs=0.0625`, `p99_abs=0.0625`, and cosine `0.9999958` under the
documented BF16 residual contract.

Source-to-installed-artifact performance parity passed over 17 public
auto-dispatch shapes: median artifact/source `0.9986`, p95 `1.0195`, and max
`1.0244`.
Comparisons against the original FlashRT pointer entry are reported separately
in `benchmarks/RESULTS.md`.

The final clean local artifact was built from
`d31c69b1cb97ecd703aba01e29f423097f11c86a`. All 17 production rows passed the
dispatcher gate; the worst auto/fastest-valid-tile paired ratio was `1.0028`.
Sixteen rows matched the original CUTLASS 4.4.2 native entry within about 1.3%
in the paired graph comparison. The PI0.5 gate/up row is a documented CUTLASS
4.5.2 version outlier at `1.128x`; it is not described as native-performance
parity.

Before the SM110 update was published, the existing Thor pipeline dependency
set was cold-loaded from Hub using both `kernels==0.16.0` and
`kernels==0.12.3`: 20/20 package imports passed for each client. The Thor host
required `HF_ENDPOINT=https://hf-mirror.com`; direct access to
`huggingface.co` timed out, so official-endpoint cold loading remains a
post-publication check on a host with direct Hub access.

## HF Jobs Publish Status

`flashrt/fp8-gemm` v1 was built and uploaded through the repository HF Jobs
workflow.

- Hub revision checked on June 20, 2026: `166f09be`
- Uploaded variants:
  - `torch211-cxx11-cu128-x86_64-linux`
  - `torch211-cxx11-cu130-x86_64-linux`
  - `torch212-cxx11-cu130-x86_64-linux`
  - `torch212-cxx11-cu132-x86_64-linux`

The existing SM120 Hub variants remain published. The new
`torch211-cxx11-cu130-aarch64-linux` SM110 artifact is not included in the
older Hub revision above; it must be published only after the clean-commit
artifact rebuild and cold-cache checks.

## SM89 Increment

The source now also exposes block-128 scaled FP8 GEMM/GEMV and
`fp8_blockwise_swiglu_quantize_fp8` on SM89. The fused producer performs the
gate/up GEMMs, SiLU product, and block-128 FP8 requantization in one launch.
It requires `1<=M<=256`, `N%128==0`, and `K%128==0` and uses the upstream
measured `32x128-w4-s1` tile. SM120 source regression remains 14/14. SM89
installed correctness, tile parity, and performance claims remain gated on an
SM89 release artifact run; source presence alone is not recorded as runtime
validation.

## SM110 portable SIMT blockwise fallback

`fp8_blockwise_linear_bf16` also ships a pure-SIMT reference
(`portable_fp8_blockwise_simt.cu`) compiled for `sm_110a`, the only arch
without a native blockwise backend. The correctness test forces the SIMT path
with `FLASHRT_FORCE_SIMT=1` and compares against the native SM120 output within
the same tolerance envelope, validating the fallback on any device. Runtime
validation of the SM110 artifact on Thor hardware is pending and is not
recorded as passed here.
