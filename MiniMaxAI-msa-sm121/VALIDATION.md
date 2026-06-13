# Validation

## Target

- Kernel family: MiniMax M3 sparse attention (MSA)
- Package: `flashrt/MiniMaxAI-msa-sm121`
- Target GPU: SM121 / GB10 / consumer Blackwell
- Dtype: BF16 inputs with FP32 accumulation references
- Layout: paged KV cache
- Model path: FlashRT MiniMax-Spark runtime on DGX Spark / GB10

## Correctness Gate

Run quick validation:

```bash
PYTHONPATH=MiniMaxAI-msa-sm121/torch-ext \
  python MiniMaxAI-msa-sm121/tests/test_msa_sm121.py --quick
```

Run full validation:

```bash
PYTHONPATH=MiniMaxAI-msa-sm121/torch-ext \
  python MiniMaxAI-msa-sm121/tests/test_msa_sm121.py
```

Expected full coverage:

| Area | Shapes | Reference | Required |
|---|---:|---|---|
| Decode sparse GQA attention | ctx 128, 2048, 4096, 32768 | paged FP32 PyTorch | cos >= 0.999, max_abs <= 5e-2 |
| Decode sparse GQA attention with sink | ctx 2048, 32768 | paged FP32 PyTorch | cos >= 0.999, max_abs <= 5e-2 |
| Decode lightning indexer | ctx 2048, 4096, 32768 | PyTorch blockmax top-k set | overlap >= 0.99 |

## FlashRT Integration Note

FlashRT has validated the decode sparse path on SM121 over context lengths
128 to 32768 with cosine similarity >= 0.999. The package tests are standalone
and do not require FlashRT, SGLang, or vLLM.

The same decode sparse path has also been exercised in FlashRT's MiniMax-Spark
model runtime on DGX Spark / GB10. That end-to-end validation is intentionally
kept as a FlashRT runtime validation item, while this Hub package exposes the
standalone kernel API for community use.

## SM121 Package Validation

Remote SM121 validation environment:

| Field | Value |
|---|---|
| Host | `spark-f517` |
| GPU | NVIDIA GB10 |
| Compute capability | 12.1 |
| Driver | 580.159.03 |
| Python | 3.12.3 |
| PyTorch | 2.12.0+cu130 |
| Triton | 3.7.0 |

Command:

```bash
PY=/home/leadtek/jax/bin/python
PYTHONPATH=MiniMaxAI-msa-sm121/torch-ext \
  $PY MiniMaxAI-msa-sm121/tests/test_msa_sm121.py
```

Result:

| Check | Shape | Cosine | Max abs / overlap | Verdict |
|---|---|---:|---:|---|
| Decode sparse GQA | ctx128_b1 | 0.999998 | 1.6032e-03 | PASS |
| Decode sparse GQA | ctx2048_b1 | 0.999996 | 4.9090e-04 | PASS |
| Decode sparse GQA | ctx2048_b2_sink | 0.999996 | 6.8302e-04 | PASS |
| Decode sparse GQA | ctx4096_b1 | 0.999996 | 4.5899e-04 | PASS |
| Decode sparse GQA | ctx4096_b2_mixed | 0.999996 | 7.3129e-04 | PASS |
| Decode sparse GQA | ctx32768_b1 | 0.999996 | 6.9451e-04 | PASS |
| Decode sparse GQA | ctx32768_b1_sink | 0.999996 | 5.6115e-04 | PASS |
| Decode top-k indexer | ctx2048 | n/a | overlap 1.000 | PASS |
| Decode top-k indexer | ctx4096 | n/a | overlap 1.000 | PASS |
| Decode top-k indexer | ctx32768 | n/a | overlap 1.000 | PASS |

The warning `tl.make_block_ptr is deprecated` appears with Triton 3.7.0. It is
a deprecation warning, not a correctness failure.
