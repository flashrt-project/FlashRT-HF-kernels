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
