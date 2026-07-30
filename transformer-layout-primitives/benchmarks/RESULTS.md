# transformer-layout-primitives Benchmark Results

Source-extension benchmark on local RTX 5090.

Environment:

- GPU: NVIDIA GeForce RTX 5090
- Backend: source extension
- Runtime: PyTorch 2.11 CUDA 12.8 validation environment
- Command:
  `python transformer-layout-primitives/benchmarks/benchmark.py --backend source --mode full --warmup 20 --iters 100`

Correctness gate:

- `python transformer-layout-primitives/tests/test_transformer_layout_primitives.py --backend source --mode full`
- Result: `82/82` checks passed
- Pure layout/RoPE checks: exact BF16 match
- Fused Q/K RMSNorm+RoPE: max_abs <= 0.015625, cosine >= 0.999999
- Q/K pair vs staged package kernels: exact BF16 match
- Q/K pair vs PyTorch valid-RoPE reference: max_abs <= 0.015625,
  cosine >= 0.999999

| Workload | Shape | Op | FlashRT us | PyTorch eager us | Speedup |
| --- | ---: | --- | ---: | ---: | ---: |
| gqa_prefill | `2520x8x128x4` | `repeat_interleave_heads_bf16` | 8.260 | 12.468 | 1.51x |
| decode_gqa | `1x8x128x4` | `repeat_interleave_heads_bf16` | 2.675 | 5.053 | 1.89x |
| short_prefill | `128x8x128x4` | `repeat_interleave_heads_bf16` | 2.721 | 4.966 | 1.82x |
| vl_prefill | `4096x8x128x4` | `repeat_interleave_heads_bf16` | 12.320 | 20.564 | 1.67x |
| qwen_prefill | `4096x32x128` | `qk_rmsnorm_rope_bf16_` | 128.920 | 854.504 | 6.63x |
| qwen_prefill | `4096x32x128` | `rope_rotate_half_bf16_` | 30.766 | 405.777 | 13.19x |
| video_prefill | `2520x24x128` | `qk_rmsnorm_rope_bf16_` | 62.107 | 250.638 | 4.04x |
| video_prefill | `2520x24x128` | `rope_rotate_half_bf16_` | 17.263 | 120.806 | 7.00x |
| decode | `1x32x128` | `qk_rmsnorm_rope_bf16_` | 5.682 | 81.028 | 14.26x |
| decode | `1x32x128` | `rope_rotate_half_bf16_` | 5.482 | 45.054 | 8.22x |
| short_prefill | `128x32x128` | `qk_rmsnorm_rope_bf16_` | 8.245 | 81.430 | 9.88x |
| short_prefill | `128x32x128` | `rope_rotate_half_bf16_` | 5.421 | 46.481 | 8.57x |
| text_tokens | `8x2048x2048` | `text_gather_bf16` | 4.113 | 14.284 | 3.47x |

New Q/K pair rows, measured after the 2026-07-30 source update with 30 warmup
iterations and 200 timed iterations:

| Workload | Shape | Op | FlashRT us | PyTorch eager us | Speedup |
| --- | ---: | --- | ---: | ---: | ---: |
| GROOT N1.7 LLM | `277x(16Q+8K)x128` | `qk_pair_rmsnorm_rope_bf16` | 8.216 | 166.775 | 20.30x |
| Qwen3-VL vision | `1024x(16Q+16K)x72` | `qk_pair_rmsnorm_rope_bf16` | 30.757 | 170.799 | 5.55x |
| LingBot vision | `1024x(16Q+16K)x80` | `qk_pair_rmsnorm_rope_bf16` | 31.788 | 163.784 | 5.15x |
| video transformer | `2520x(24Q+24K)x128` | `qk_pair_rmsnorm_rope_bf16` | 109.040 | 518.938 | 4.76x |
| action boundary | `51x(16Q+16K)x64` | `qk_pair_rmsnorm_rope_bf16` | 4.108 | 169.040 | 41.15x |
| Wan partial tile | `5070x(24Q+24K)x128` | `qk_pair_rmsnorm_rope_bf16` | 266.236 | 1492.024 | 5.60x |

Notes:

- These are microbenchmarks for reusable layout/math primitives, not end-to-end
  model throughput claims.
- `repeat_interleave_heads_bf16` uses a source-driven write pattern to avoid
  repeated source loads on GQA/MQA expansion.
- The Q/K fused path validates the same BF16 writeback order as the CUDA
  implementation: RMSNorm writes BF16 normalized values before RoPE.

### Q/K pair raw-launcher parity

| Shape | Native us | Wrapper us | Wrapper/native | Graph native us | Graph wrapper us | Graph wrapper/native | Eager us | Compile us | Exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| GROOT `277x(16Q+8K)x128` | 8.212 | 8.214 | 1.000 | 7.431 | 7.443 | 1.002 | 162.040 | 30.253 | yes |
| Qwen vision `1024x(16Q+16K)x72` | 30.752 | 30.752 | 1.000 | 30.266 | 30.479 | 1.007 | 190.518 | 41.705 | yes |
| LingBot `1024x(16Q+16K)x80` | 32.760 | 32.774 | 1.000 | 30.602 | 30.706 | 1.003 | 188.639 | 41.035 | yes |
| Wan `2520x(24Q+24K)x128` | 108.627 | 109.808 | 1.011 | 109.192 | 109.304 | 1.001 | 525.848 | 65.674 | yes |
| action `51x(16Q+16K)x64` | 4.111 | 4.111 | 1.000 | 2.756 | 2.755 | 1.000 | 162.765 | 42.270 | yes |

## Indexed-row native parity

The generic gather implementation was compared with the original Cosmos Edge
raw CUDA launcher on the same buffers. It is bitwise identical.

| Shape | Native us | Wrapper us | Wrapper/native | Graph native us | Graph wrapper us | Graph wrapper/native | Eager us | Compile us |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gather `2520 -> 105`, `D=1152` | 2.055 | 2.233 | 1.086 | 1.152 | 1.152 | 1.000 | 4.287 | 21.618 |
| scatter `105 -> 2520`, `D=1152` | 2.048 | 2.228 | 1.088 | 1.024 | 1.024 | 1.000 | 14.343 | 26.500 |

For this launch-floor copy primitive, CUDA Graph parity is the runtime gate;
the direct-call delta is Tensor dispatcher overhead.
