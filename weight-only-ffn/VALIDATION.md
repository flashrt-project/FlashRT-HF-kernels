# Validation

## Correctness Contract

Every compute result is compared with a staged reference built from the exact
dequantized static weight and explicit BF16 rounding at region boundaries.
The gate records:

- output dtype
- maximum, mean, and p99 absolute error
- maximum and p99 stabilized relative error
- cosine similarity

Acceptance thresholds:

- dtype: `torch.bfloat16`
- max absolute error: `<= 0.015625`
- mean absolute error: `<= 0.0005`
- p99 absolute error: `<= 0.00390625`
- p99 relative error, with denominator clamped at `0.125`: `<= 0.02`
- cosine similarity: `>= 0.9999`

The relative denominator clamp prevents values near zero from turning a small
BF16 absolute difference into a meaningless large percentage.

## Required Commands

Source implementation:

```bash
python weight-only-ffn/tests/test_weight_only_ffn.py \
  --backend source --mode full
```

Installed builder artifact:

```bash
python weight-only-ffn/tests/test_weight_only_ffn.py \
  --backend installed --mode full \
  --artifact weight-only-ffn/build/torch211-cxx11-cu128-x86_64-linux
```

The full gate verifies that W4 rejects `M=5`, W8 accepts through `M=8` and
rejects `M=9`, and known weak geometries do not silently run an unqualified
dense path. It covers the draft K envelope through `K=17408`. Installed
full mode also traces and executes the public W8A16 wrapper with
`torch.compile(fullgraph=True)` and requires exact parity with its eager call.

## Performance Gate

```bash
python weight-only-ffn/benchmarks/benchmark.py \
  --backend installed --mode full \
  --artifact weight-only-ffn/build/torch211-cxx11-cu128-x86_64-linux
```

The Qwen3.8 draft gate is separate so its large static weights do not overlap
the general full sweep in GPU memory:

```bash
python weight-only-ffn/benchmarks/benchmark.py \
  --backend installed --mode draft \
  --artifact weight-only-ffn/build/torch213-cxx11-cu130-x86_64-linux
```

For each FC/QKV/O/gate-up/down family it requires M=1 to reach at least 1.6x
versus BF16 eager GEMV and M=8 latency to remain within 2x of M=1.

Each op is measured against both PyTorch eager and a warmed
`torch.compile(mode="max-autotune-no-cudagraphs")` reference. Variant timings
are retained so the selected auto dispatch can be audited. An accepted auto
path fails the benchmark if it is more than 5% slower than the fastest
diagnostic tile or if it does not beat the stronger eager/compile baseline by
at least 2%. Rejected shapes remain visible as `auto_status="rejected"` and are
never reported as production speedups. The full matrix contains complete FFN
regions and standalone linear projections so a fast first projection cannot
hide a weak second projection.

## RTX 5090 Release-Candidate Evidence

The source release candidate was tested on an NVIDIA GeForce RTX 5090 with
Torch 2.9.1+cu128:

- correctness: `26/26` checks passed
- worst W8 cosine similarity: `0.9999913`
- worst W8 p99 absolute error: `0.001953125`
- public W8 linear wrapper: exact eager/`torch.compile(fullgraph=True)` parity
- performance sweep: 76 rows, 51 accepted and 25 explicitly rejected
- minimum accepted speedup: `1.22x` versus eager and `1.37x` versus compile
- maximum auto-to-best-diagnostic-tile gap: `2.82%`

The release flake pins upstream kernel-builder commit
`19aaa6421e674e9fecc352bbae6eab81d19a6bf4`. With CUDA 12.8+ filtering, the
expected x86_64 release matrix is Torch 2.11 cu128/cu130, Torch 2.12
cu130/cu132, and Torch 2.13 cu130/cu132. HF Jobs must build and upload every
eligible variant before the Hub release is considered complete.

## Thor SM110 Evidence

The CUDA 13 builder-generated aarch64 artifact was tested on NVIDIA Thor,
SM110, with Torch 2.11.0+cu130:

- source correctness: 26/26 checks passed;
- installed-artifact correctness: 26/26 checks passed;
- generated fatbin contains SM110a, SM120a, and SM121 cubins, while the CUDA
  12.8 SM120 component remains unchanged;
- complete FFN sweep: W4 accepts 8 winning rows and W8 accepts 21 winning
  rows; the standalone linear sweep accepts two W4 and two W8 rows; zero
  accepted rows lose to the stronger eager/compile baseline;
- minimum complete-region speedup: W4 `1.188x`, W8 `1.352x` against the
  stronger baseline;
- accepted auto selection is at most `0.76%` from the fastest diagnostic tile;
- standalone projection sweep covers eight model-oriented shapes for both W4
  and W8. Thor auto dispatch retains only the large wide-projection envelope
  and rejects square, down-projection, and small-weight rows that lose to
  eager BF16 GEMM.
