# Benchmark Results

Initial package validation focuses on correctness and SM121 enablement.
Numbers below are from the v1 source-level Triton path; v2 native-helper
artifact benchmark must be refreshed after HF Jobs publishes the v2 build.

Environment:

| Field | Value |
|---|---|
| Host | `spark-f517` |
| GPU | NVIDIA GB10 |
| Compute capability | 12.1 |
| Driver | 580.159.03 |
| Python | 3.12.3 |
| PyTorch | 2.12.0+cu130 |
| Triton | 3.7.0 |

Source-tree command:

```bash
PY=/home/leadtek/jax/bin/python
PYTHONPATH=MiniMaxAI-msa-sm121/torch-ext \
  $PY MiniMaxAI-msa-sm121/benchmarks/benchmark_decode.py \
    --ctx 2048 4096 32768 --warmup 3 --iters 10
```

Source-level decode sparse benchmark:

| Context | Attention mean us | Native top-k mean us |
|---:|---:|---:|
| 2048 | 60.292 | n/a |
| 4096 | 57.642 | n/a |
| 32768 | 58.654 | n/a |

These are source-level package smoke benchmarks on SM121. Public performance
claims should be refreshed from the uploaded v2 Hub artifact after HF Jobs
publish. In source-tree mode the native CUDA extension is not built, so the
native top-k helper column is intentionally `n/a`.
