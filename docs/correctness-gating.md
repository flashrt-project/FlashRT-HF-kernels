# Correctness Gating

Correctness is the first release gate. A kernel package cannot enter the v1
release build window just because it has a benchmark script or a local smoke
run. Every output that the public API writes must be validated before timing
results are accepted.

## Gate Levels

| Level | Meaning | Release use |
| --- | --- | --- |
| C0 | Source compiles or launches | Not enough |
| C1 | One smoke shape passes | Source sync only |
| C2 | Representative shape family passes | Draft package |
| C3 | Full v1 shape grid passes all public outputs | Release candidate |
| C4 | Full v1 grid passes on every advertised hardware scope | Community-ready |

## Rules

- Accuracy is measured before latency.
- Every public output tensor must be validated. For example, a Q/K API must
  validate both Q and K.
- Multi-output kernels cannot rely on the HF benchmark runner alone if the
  runner verifies only one tensor. Use package tests or an internal accuracy
  sweep for all outputs.
- Benchmark tables must not include speedups for shapes that fail the declared
  accuracy threshold.
- Max error alone is not enough for approximate kernels. Record max absolute
  error, max relative error, and an error distribution summary.
- Exact byte-output kernels must use byte parity unless the API explicitly
  documents an approximate output contract.

## Current V1 Correctness Status

| Package | Current correctness gate | Status |
| --- | --- | --- |
| `flashrt-gemm-epilogues` | C2 partial | FP8 quant epilogues have exact FP8 parity tests. BF16 GEMM epilogue wrappers use loose BF16 tolerances and are not headline-ready. |
| `flashrt-vla-video` | C1/C2 mixed | Q/K single-output primitives have local smoke evidence. QKV split + norm + RoPE is blocked until Q and K both pass an accuracy-first sweep. |
| `flashrt-nvfp4` | C2 partial | Layout helper has byte-parity tests on representative shapes; full benchmark grid must be verified before release results. |
| `flashrt-smallm-gemm` | C1 | Deterministic constant-input smoke passes for `K=4096,12288`, but the full `N` grid and random/dequantized references are not complete. |
| `flashrt-fused-quant` | C1 | Split/merged byte parity smoke passes on small shapes, but the full v1 shape grid is not complete. |

## Package Requirements Before V1 Build Window

### `flashrt-gemm-epilogues`

- FP8 quant APIs: exact FP8 byte parity across the v1 shape grid.
- BF16 GEMM epilogues: either a stronger reference and justified tolerance, or
  keep them as compatibility APIs without headline claims.

### `flashrt-vla-video`

- `q_norm_rope_bf16`: Q output accuracy across head-count grid.
- `k_norm_rope_v_cache_bf16`: K output accuracy and exact V copy across
  head-count grid.
- `qkv_split_norm_rope_bf16`: Q and K output accuracy across token and
  head-count grids.
- No QKV speedup table is accepted until Q and K both pass.

### `flashrt-nvfp4`

- `nvfp4_sf_linear_to_swizzled`: byte parity across every v1 layout shape.
- Caller-provided output reuse must preserve byte parity.

### `flashrt-smallm-gemm`

- `nvfp4_w4a4_decode_matvec_bf16out`: deterministic and random/dequantized
  references across `K in {4096,12288}` and `N in {1024,4096,12288}`.
- Unsupported shape rejection must be tested.

### `flashrt-fused-quant`

- Split and merged `SiLU(gate) * up -> NVFP4` must pass packed output and
  swizzled scale byte parity across the v1 decode, small-batch, prefill, and
  video shape grid.
- Caller-provided output buffers must be tested, including scale padding
  behavior.

## Build Policy

Run:

```bash
python scripts/correctness_audit.py
```

The v1 full build window must not start while this audit reports blockers.
