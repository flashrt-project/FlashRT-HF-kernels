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
- Result: `32/32` checks passed
- Pure layout/RoPE checks: exact BF16 match
- Fused Q/K RMSNorm+RoPE: max_abs <= 0.0078125, cosine >= 0.999999

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

Notes:

- These are microbenchmarks for reusable layout/math primitives, not end-to-end
  model throughput claims.
- `repeat_interleave_heads_bf16` uses a source-driven write pattern to avoid
  repeated source loads on GQA/MQA expansion.
- The Q/K fused path validates the same BF16 writeback order as the CUDA
  implementation: RMSNorm writes BF16 normalized values before RoPE.
