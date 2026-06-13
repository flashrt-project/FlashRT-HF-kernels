# Benchmarks

Benchmark scripts are source-level helpers. Public claims should be recorded
only after running on the target GPU and exact package artifact.

```bash
PYTHONPATH=MiniMaxAI-msa-blackwell/torch-ext \
  python MiniMaxAI-msa-blackwell/benchmarks/benchmark_decode.py \
    --ctx 2048 4096 32768 65536 131072
```

The script reports two columns:

- `attention_mean_us`: sparse GQA decode attention path.
- `native_topk_mean_us`: native CUDA score-to-top-k helper. It is `NA` in
  source-tree mode before a built artifact is installed.
