# Benchmark Results: flashrt-vla-video

This file is the public result ledger for the v1 VLA/video block. The previous
local QKV split + norm + RoPE speedup table is invalidated as release evidence:
it reported max absolute errors up to `0.25` without a full accuracy
explanation, and the public HF benchmark script verified only one output tensor.

Do not use this package for a public speedup claim until the accuracy-first
benchmark gate below is completed.

Validated on June 2, 2026.

Environment:

- GPU: NVIDIA GeForce RTX 5090
- PyTorch: 2.9.1+cu128
- CUDA runtime reported by PyTorch: 12.8
- Timing path: FlashRT internal pybind module for source selection, followed by
  a package-local source-extension smoke benchmark.
- Tile sweep: warmup 20, measured iterations 100.
- Tile override: `FLASHRT_QKV_ROPE_BLOCK_SIZE in {128, 256, 512}`.

## Current Status

- `q_norm_rope_bf16` and `k_norm_rope_v_cache_bf16` have local correctness
  smoke evidence, but still need built-artifact HF benchmark results.
- `qkv_split_norm_rope_bf16` remains a candidate fused post-processing API, but
  its previous performance table is not valid release evidence.
- No VLA/video headline speedup is approved in this ledger yet.

## Accuracy Gate Required

Before restoring any QKV split + norm + RoPE speedup table, record:

- Q and K output validation, not only Q.
- `max_abs_error`, `max_rel_error`, and error distribution for every shape.
- A clear BF16 reference policy: exact BF16 operation order, FP32 reference, or
  model-tolerance reference.
- Pass/fail threshold chosen before timing is reported.
- Timing results only for shapes that pass the accuracy threshold.

## Tile Policy

The current SM120 default policy is:

- use 512-thread CTAs for `tokens <= 64`;
- use 256-thread CTAs for longer token blocks.

The policy favors short-context and head-count sweep wins while avoiding the
long-token regression seen with 512-thread CTAs at `tokens >= 2520`.

## Q RMSNorm + RoPE + Stage Write

| n_heads | Fused us | PyTorch eager us | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 2.233 | 76.245 | 34.14x |
| 4 | 2.284 | 66.647 | 29.18x |
| 8 | 2.062 | 66.312 | 32.16x |
| 16 | 2.078 | 66.409 | 31.96x |
| 32 | 2.212 | 66.513 | 30.08x |
| 48 | 2.194 | 66.515 | 30.32x |

## K RMSNorm + RoPE + K-Cache Write + V-Cache Copy

| n_heads | Fused us | PyTorch eager us | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 2.062 | 68.507 | 33.22x |
| 4 | 2.075 | 69.296 | 33.39x |
| 8 | 2.154 | 69.255 | 32.15x |
| 16 | 2.313 | 69.502 | 30.05x |
| 32 | 2.062 | 70.425 | 34.15x |
| 48 | 2.075 | 70.232 | 33.85x |

## Package-Local Source Smoke

The package-local source extension was compiled with
`torch.utils.cpp_extension.load` using:

- `torch-ext/torch_binding.cpp`
- `csrc/q_norm_rope_bf16.cu`

| Shape | Q fused us | Q eager us | Q speedup | K fused us | K eager us | K speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| heads=1 | 2.639 | 71.105 | 26.95x | 2.564 | 73.607 | 28.71x |
| heads=8 | 2.464 | 71.944 | 29.20x | 2.555 | 74.246 | 29.06x |
| heads=48 | 2.454 | 75.614 | 30.81x | 2.667 | 77.466 | 29.05x |

## Invalidated QKV Sweep

The previous package-local QKV sweep is intentionally removed from this public
ledger. The highest recorded max absolute error was `0.25`, and the benchmark
path did not verify both Q and K outputs through the HF benchmark runner. Keep
those numbers as internal debugging context only.

## Pending Release Results

Run after the accuracy gate and built package artifact both pass:

```bash
kernels benchmark flashrt/flashrt-vla-video \
  --benchmark-script benchmarks/benchmark_q_norm_rope.py
```

Record:

| Workload | Shape | Mean ms | Ref ms | Speedup | Max abs err | Max rel err | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| pending | pending | pending | pending | pending | pending | pending | Accuracy-first benchmark not run yet |
