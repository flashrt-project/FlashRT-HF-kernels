# Validation

Correctness smoke:

```bash
python flashrt-flex-attention-train/tests/test_flashrt_flex_attention_train.py --backend source --mode smoke
python flashrt-flex-attention-train/tests/test_flashrt_flex_attention_train.py --backend source --mode full
```

Installed artifact smoke:

```bash
python flashrt-flex-attention-train/tests/test_flashrt_flex_attention_train.py --backend installed --mode full
python scripts/prebuild_check.py --package flashrt-flex-attention-train --check-config
```

Microbenchmark and acceptance gates:

```bash
python flashrt-flex-attention-train/benchmarks/benchmark.py --device cuda --dtype bf16 --mode all --output /tmp/flex_attention_a100.json
python flashrt-flex-attention-train/benchmarks/benchmark.py --device cuda --dtype bf16 --mode all --require-gates
```

Shape/tile matrix:

```bash
python flashrt-flex-attention-train/benchmarks/shape_matrix.py --presets a100 --output /tmp/a100_flex_matrix.jsonl
python flashrt-flex-attention-train/benchmarks/shape_matrix.py --presets consumer --output /tmp/consumer_flex_matrix.jsonl
```

Minimum gate for connecting native kernels:

- forward time <= `0.95 * SDPA`
- forward+backward time <= `0.95 * SDPA`
- peak memory <= `1.03 * SDPA`
- forward max abs diff <= `2e-3`
- gradient norm relative diff <= `1e-2`

Higher gate for publishing native kernels:

- full PI052 step speedup >= `1.08x` on A100 and RTX 5090
- isolated attention forward+backward speedup >= `1.25x`
- no text/flow step regression
- unsupported shapes automatically fall back to SDPA

The current package provides the reference/fallback path and therefore is not
expected to pass the performance gates until native CUDA fwd/bwd kernels are
added behind the same API.
