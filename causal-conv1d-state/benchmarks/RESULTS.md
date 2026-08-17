# Results

## RTX 5090 Source Sanity Benchmark

Command:

```bash
python causal-conv1d-state/benchmarks/benchmark.py \
  --backend source \
  --mode headline \
  --iters 200 \
  --json-out internal-tests/causal-conv1d-state-source-benchmark.json
```

| Shape | Kernel us | PyTorch reference us | Notes |
| --- | ---: | ---: | --- |
| parallel_s16_c1024 | 9.075 | 1768.035 | Python/Torch state reference, sanity only |
| gqa_s8_c10240 | 11.178 | 787.005 | Python/Torch state reference, sanity only |

These rows are source-extension sanity numbers. The reference is a simple
Python/Torch state contract and should not be used as the public competitive
baseline. Built-artifact and runtime-pipeline benchmarks should be regenerated
after HF Jobs upload succeeds.

## Step-batched GQA prefill

RTX 5090 source artifact, PyTorch `2.11.0+cu128`, `S=2044`, `C=10240`, `K=4`,
caller-owned outputs and in-place state:

| Entry | Latency | Relative |
| --- | ---: | ---: |
| `causal_conv1d_update_steps_gqa_bf16` | 45.44 us | 2.67x faster |
| existing chunk-parallel GQA entry | 121.22 us | 1.00x |

The full gate covers `S=1/63/64/2044`. At `S=2044`, max absolute error is
`0.000244`, p99 is zero, cosine is `1.0`, and final state is exact.
