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

The full gate also verifies that default dispatch rejects `M=5` and known weak
small-M geometries rather than running an unqualified dense path. Installed
full mode also traces and executes the public W8A16 wrapper with
`torch.compile(fullgraph=True)` and requires exact parity with its eager call.

## Performance Gate

```bash
python weight-only-ffn/benchmarks/benchmark.py \
  --backend installed --mode full \
  --artifact weight-only-ffn/build/torch211-cxx11-cu128-x86_64-linux
```

Each op is measured against both PyTorch eager and a warmed
`torch.compile(mode="max-autotune-no-cudagraphs")` reference. Variant timings
are retained so the selected auto dispatch can be audited. An accepted auto
path fails the benchmark if it is more than 5% slower than the fastest
diagnostic tile. Rejected shapes remain visible as `auto_status="rejected"` and
are never reported as production speedups.

## RTX 5090 Release-Candidate Evidence

The installed `torch211-cxx11-cu128-x86_64-linux` kernel-builder artifact was
tested on an NVIDIA GeForce RTX 5090 with Torch 2.11.0+cu128:

- correctness: `26/26` checks passed
- worst W8 cosine similarity: `0.9999913`
- worst W8 p99 absolute error: `0.001953125`
- public W8 linear wrapper: exact eager/`torch.compile(fullgraph=True)` parity
- performance sweep: 60 rows, 39 accepted and 21 explicitly rejected
- minimum accepted speedup: `1.22x` versus eager and `1.38x` versus compile
- maximum auto-to-best-diagnostic-tile gap: `1.81%`

The release flake pins upstream kernel-builder commit
`19aaa6421e674e9fecc352bbae6eab81d19a6bf4`. With CUDA 12.8+ filtering, the
expected x86_64 release matrix is Torch 2.11 cu128/cu130, Torch 2.12
cu130/cu132, and Torch 2.13 cu130/cu132. HF Jobs must build and upload every
eligible variant before the Hub release is considered complete.
