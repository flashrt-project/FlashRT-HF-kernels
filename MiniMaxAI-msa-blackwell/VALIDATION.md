# Validation

## Target

- Kernel family: MiniMax M3 sparse attention (MSA)
- Package: `flashrt/MiniMaxAI-msa-blackwell`
- HF Jobs package selector: `MiniMaxAI-msa-blackwell`
- Package version: v1 Blackwell native-helper package
- Target GPU family: Blackwell CUDA compute capability 12.x
- Validated GPU: SM121 / GB10 / DGX Spark
- Dtype: BF16 inputs with FP32 accumulation references
- Layout: paged KV cache
- Model path: FlashRT MiniMax-Spark runtime on DGX Spark / GB10

## Correctness Gate

Run quick validation:

```bash
PYTHONPATH=MiniMaxAI-msa-blackwell/torch-ext \
  python MiniMaxAI-msa-blackwell/tests/test_msa_blackwell.py --quick
```

Run full validation:

```bash
PYTHONPATH=MiniMaxAI-msa-blackwell/torch-ext \
  python MiniMaxAI-msa-blackwell/tests/test_msa_blackwell.py
```

Run standalone long-context validation:

```bash
PYTHONPATH=MiniMaxAI-msa-blackwell/torch-ext \
  python MiniMaxAI-msa-blackwell/tests/test_msa_blackwell.py --long-context
```

Expected full coverage:

| Area | Shapes | Reference | Required |
|---|---:|---|---|
| API surface | official `MiniMaxAI/msa` public names | `api_status.py` | all official root names exported; no unsupported public root API entries |
| Native CUDA top-k helper | heads 64, batch 1-2, blocks 1-256 | PyTorch top-k over valid blocks | exact set match |
| Decode sparse GQA attention | ctx 128, 2048, 4096, 32768 | paged FP32 PyTorch | cos >= 0.999, max_abs <= 5e-2 |
| Prefill sparse GQA attention | ctx 512, 4096 | paged causal FP32 PyTorch | cos >= 0.999, max_abs <= 5e-2 |
| Decode sparse GQA attention with sink | ctx 2048, 32768 | paged FP32 PyTorch | cos >= 0.999, max_abs <= 5e-2 |
| Official decode API wrapper | ctx 2048, 4096 | direct Blackwell decode kernel | cos = 1.0, max_abs = 0 |
| Official CSR prefill API wrapper | ctx 512, 2048 | direct Blackwell prefill kernel | cos = 1.0, max_abs = 0 under CSR-preserved block order |
| Official NVFP4 prefill API wrapper | ctx 512 BF16 fallback path | `sparse_atten_func` | cos = 1.0, max_abs = 0 |
| Official FP4 indexer API fallback | tiny FP4 packed tensors | shape/finite-score check | returns official score layout |
| Decode lightning indexer | ctx 2048, 4096, 32768 | PyTorch blockmax top-k set | overlap >= 0.99 |
| Standalone long-context decode | ctx 65536, 131072 | paged FP32 PyTorch / direct kernel | cos >= 0.999; wrapper max_abs = 0 |
| Installed-artifact native long top-k | blocks 512, 1024 | PyTorch top-k over valid blocks | exact set match |

API surface validation:

```bash
PYTHONPATH=MiniMaxAI-msa-blackwell/torch-ext \
  python -m pytest MiniMaxAI-msa-blackwell/tests/test_api_surface.py -q
```

The test tracks every official `MiniMaxAI/msa` public API name:

- `sparse_atten_func`
- `sparse_atten_nvfp4_kv_func`
- `sparse_decode_atten_func`
- `SparseDecodePagedAttentionWrapper`
- `fp4_indexer_block_scores`
- `build_k2q_csr`
- `SparseK2qCsrBuilderSm100`
- `Nvfp4QuantizedTensor`
- `quantize_bf16_to_nvfp4_128x4`
- `quantize_kv_bf16_to_nvfp4_128x4`
- `dequantize_nvfp4_128x4_to_bf16`
- `swizzle_nvfp4_scale_to_128x4`
- `nvfp4_global_scale_from_amax`

The root module exports every official public name. Decode, CSR prefill, NVFP4
prefill compatibility, FP4 block scoring, CSR, and NVFP4 helper names are all
callable. The optimized SM100 CUTE bodies are not claimed as ported here; where
that matters, this package uses Blackwell Triton kernels or correctness-first
fallbacks instead of returning fake results.

## FlashRT Integration Note

FlashRT has validated the decode sparse path on SM121 over context lengths
128 to 32768 with cosine similarity >= 0.999. The 32768 context length has
also been exercised in the FlashRT MiniMax-Spark model runtime on DGX Spark /
GB10, so it is the current end-to-end model validation boundary.

The standalone package kernel tests additionally cover 65536 and 131072
context lengths. These long-context rows validate the kernel and API wrapper
contract outside the full model runtime; they should not be described as
MiniMax-Spark end-to-end model validation until the full runtime path is rerun
at those lengths.

The same decode sparse path has also been exercised in FlashRT's MiniMax-Spark
model runtime on DGX Spark / GB10. That end-to-end validation is intentionally
kept as a FlashRT runtime validation item, while this Hub package exposes the
standalone kernel API for community use.

## Native Helper Compile Smoke

Before HF Jobs publish, the native helper was compiled locally as a PyTorch
extension using the same source files:

- `torch-ext/torch_binding.cpp`
- `csrc/msa_topk_from_scores.cu`

Environment:

| Field | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 |
| PyTorch | 2.9.1+cu128 |
| nvcc | CUDA 13.0 |
| Target arch | sm_120 |

Result:

| Check | Shape | Reference | Verdict |
|---|---:|---|---|
| Native score -> top-k | heads 64, batch 1, blocks 256, topk 16 | PyTorch top-k set | PASS |

## Blackwell Package Validation

Remote Blackwell validation environment:

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
PYTHONPATH=MiniMaxAI-msa-blackwell/torch-ext \
  $PY MiniMaxAI-msa-blackwell/tests/test_msa_blackwell.py
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
| Decode sparse GQA | ctx65536_b1 | 0.999996 | 4.3470e-04 | PASS |
| Decode sparse GQA | ctx131072_b1 | 0.999996 | 7.1825e-04 | PASS |
| Decode top-k indexer | ctx2048 | n/a | overlap 1.000 | PASS |
| Decode top-k indexer | ctx4096 | n/a | overlap 1.000 | PASS |
| Decode top-k indexer | ctx32768 | n/a | overlap 1.000 | PASS |
| Decode top-k indexer | ctx65536 | n/a | overlap 1.000 | PASS |
| Decode top-k indexer | ctx131072 | n/a | overlap 1.000 | PASS |
| Official decode wrapper | ctx2048 | 1.000000 | 0.0000e+00 | PASS |
| Official decode wrapper | ctx4096 | 1.000000 | 0.0000e+00 | PASS |
| Official decode wrapper | ctx65536 | 1.000000 | 0.0000e+00 | PASS |
| Official decode wrapper | ctx131072 | 1.000000 | 0.0000e+00 | PASS |

Installed-artifact native top-k validation on RTX 5090 / torch 2.11 / CUDA
12.8:

| Context | Blocks | Overlap | Verdict |
|---:|---:|---:|---|
| 32768 | 256 | 1.000 | PASS |
| 65536 | 512 | 1.000 | PASS |
| 131072 | 1024 | 1.000 | PASS |

The warning `tl.make_block_ptr is deprecated` appears with Triton 3.7.0. It is
a deprecation warning, not a correctness failure.

## Native/CUTE Alignment Status

The upstream `MiniMaxAI/msa` package is an SM100 package with native helper ops
and CUTE-DSL attention kernels. This Blackwell package is being upgraded in stages:

1. Initial source-level Triton CUDA decode path.
2. Current v1 package: native CUDA score-to-top-k helper plus Blackwell Triton
   decode attention.
3. planned: Blackwell CUTE/native decode attention path aligned with upstream
   `MiniMaxAI/msa` public APIs.

Do not describe this package as a full native-CUTE attention replacement. It is
a real native package, but the attention body still uses the Blackwell Triton
fallback.
