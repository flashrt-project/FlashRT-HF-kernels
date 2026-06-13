# Benchmark Results

Initial package validation focuses on correctness and Blackwell enablement.
Numbers below are source-level smoke benchmark data. Installed-artifact
benchmark data should be refreshed after HF Jobs publishes the v1 Blackwell
package.

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
PY=python
PYTHONPATH=MiniMaxAI-msa-blackwell/torch-ext \
  $PY MiniMaxAI-msa-blackwell/benchmarks/benchmark_decode.py \
    --ctx 2048 4096 32768 --warmup 3 --iters 10
```

Source-level decode sparse benchmark:

| Context | Attention mean us | Native top-k mean us |
|---:|---:|---:|
| 2048 | 60.292 | n/a |
| 4096 | 57.642 | n/a |
| 32768 | 58.654 | n/a |

These are source-level package smoke benchmarks on GB10 / SM121. Public
performance claims should be refreshed from the uploaded v1 Hub artifact after
HF Jobs publish. In source-tree mode the native CUDA extension is not built, so
the native top-k helper column is intentionally `n/a`.

Long-context correctness has since been extended to standalone kernel rows at
65536 and 131072 context length. Performance tables for those rows should be
refreshed from the installed Hub artifact before quoting latency numbers:

```bash
python MiniMaxAI-msa-blackwell/benchmarks/benchmark_decode.py \
  --ctx 2048 4096 32768 65536 131072 --warmup 10 --iters 50
```
