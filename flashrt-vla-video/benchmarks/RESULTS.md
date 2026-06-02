# Benchmark Results: flashrt-vla-video

This file is the public result ledger for the v1 VLA/video block. The previous
local QKV split + norm + RoPE speedup table is invalidated as release evidence:
it reported max absolute errors up to `0.25` without a full accuracy
explanation, and the public HF benchmark script verified only one output tensor.

Do not use the old QKV speedup numbers for a public speedup claim. Source
correctness has been revalidated with the fixed reference. The current table
below is the built-artifact release-candidate result.

Validated on June 2, 2026.

Environment:

- GPU: NVIDIA GeForce RTX 5090
- Driver: 580.82.07
- Built artifact: `torch211-cxx11-cu128-x86_64-linux`
- PyTorch inside HF testshell: 2.11.0+cu128
- CUDA runtime reported by PyTorch: 12.8
- Local runner: `scripts/run_built_artifact_benchmarks.py`
- Timing: warmup 10, measured iterations 50.

## Current Status

- `q_norm_rope_bf16`, `k_norm_rope_v_cache_bf16`, and
  `qkv_split_norm_rope_bf16` pass built-artifact benchmark verification.
- QKV benchmark verifies Q and K through separate benchmark classes.
- Current built-artifact speedup range: 9.79x to 29.30x against the PyTorch
  eager references in `benchmark_q_norm_rope.py`.

## Source Accuracy Gate

Command:

```bash
python scripts/accuracy_sweep.py --backend source --mode full --package flashrt-vla-video
```

Result: passed 110 checks.

Covered:

- Q/K rows `1,4,8,16,24,32,48,64,128,256`.
- QKV tokens `1,4,16,64,256,1024,2520,4096`.
- QKV heads `8,16,24,32,48`, `head_dim=128`.

Accuracy contract:

- BF16 Q/K outputs: `max_abs <= 0.03125`, `max_rel <= 0.05` with
  `rel_floor=1`.
- V copy output: byte parity.
- Worst recorded QKV max absolute error: `0.015625`.

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

## Built Artifact Release-Candidate Results

Command:

```bash
python scripts/run_built_artifact_benchmarks.py \
  --package flashrt-vla-video --warmup 10 --iterations 50
```

Q norm + RoPE:

| Workload | Mean us | Ref us | Speedup | Verified |
| --- | ---: | ---: | ---: | --- |
| `heads1` | 7.48 | 76.00 | 10.16x | yes |
| `heads4` | 7.60 | 97.73 | 12.86x | yes |
| `heads8` | 7.60 | 86.96 | 11.45x | yes |
| `heads16` | 7.78 | 94.55 | 12.15x | yes |
| `heads24` | 7.66 | 78.25 | 10.21x | yes |
| `heads32` | 7.54 | 79.00 | 10.47x | yes |
| `heads48` | 7.60 | 80.01 | 10.53x | yes |

K norm + RoPE + V cache copy:

| Workload | Mean us | Ref us | Speedup | Verified |
| --- | ---: | ---: | ---: | --- |
| `heads1` | 7.49 | 75.15 | 10.04x | yes |
| `heads4` | 7.55 | 79.75 | 10.56x | yes |
| `heads8` | 7.58 | 78.19 | 10.32x | yes |
| `heads16` | 7.54 | 77.46 | 10.28x | yes |
| `heads24` | 7.46 | 76.86 | 10.31x | yes |
| `heads32` | 7.62 | 78.69 | 10.32x | yes |
| `heads48` | 7.48 | 78.53 | 10.50x | yes |

Packed QKV split + Q norm + RoPE:

| Workload | Mean us | Ref us | Speedup | Verified |
| --- | ---: | ---: | ---: | --- |
| `tokens1` | 9.87 | 145.97 | 14.79x | yes |
| `tokens4` | 10.01 | 150.44 | 15.02x | yes |
| `tokens16` | 10.09 | 145.50 | 14.43x | yes |
| `tokens64` | 10.16 | 144.85 | 14.26x | yes |
| `tokens256` | 11.63 | 145.50 | 12.51x | yes |
| `tokens1024` | 17.36 | 211.38 | 12.18x | yes |
| `tokens2520` | 26.77 | 536.17 | 20.03x | yes |
| `tokens4096` | 43.76 | 1253.80 | 28.65x | yes |

Packed QKV split + K norm + RoPE:

| Workload | Mean us | Ref us | Speedup | Verified |
| --- | ---: | ---: | ---: | --- |
| `tokens1` | 9.99 | 143.60 | 14.37x | yes |
| `tokens4` | 10.03 | 149.17 | 14.87x | yes |
| `tokens16` | 10.09 | 145.19 | 14.39x | yes |
| `tokens64` | 10.16 | 144.38 | 14.21x | yes |
| `tokens256` | 11.79 | 147.37 | 12.50x | yes |
| `tokens1024` | 17.60 | 211.54 | 12.02x | yes |
| `tokens2520` | 26.74 | 532.23 | 19.90x | yes |
| `tokens4096` | 42.55 | 1248.21 | 29.33x | yes |
