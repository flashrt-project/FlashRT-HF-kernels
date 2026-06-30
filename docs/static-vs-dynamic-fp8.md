# Static vs Dynamic FP8 Reproduction

Measured on RTX 5090 with `pi05_libero_pytorch`, public FlashRT Hugging Face
Kernel Hub packages, and the committed PI0.5 LIBERO frame-50 bundle.

Static means FP8 activation scales are calibrated once and frozen. Dynamic means
FP8 activation scales are recomputed per forward with `amax`.

## Results

### Strict Full All-FP8 E2E

This run covers vision, encoder, and decoder. `--fp8-projections` is enabled.
The dynamic side recomputes activation scales in all FP8 projection/FFN sites.

| full all-FP8 e2e | STATIC | DYNAMIC | static is |
| --- | ---: | ---: | ---: |
| CUDA graph | 21.56 ms | 36.23 ms | 1.68x faster |
| eager | 24.85 ms | 47.97 ms | 1.93x faster |
| first call | 103.5 ms | 130.5 ms | 1.26x faster |

| accuracy vs BF16 reference | STATIC | DYNAMIC |
| --- | ---: | ---: |
| max abs | 0.01953125 | 0.021484375 |
| mean abs | 0.00259846 | 0.00284873 |
| p99 abs | 0.015625 | 0.01953125 |
| MSE | 1.675e-05 | 2.008e-05 |
| action cosine | 0.9999275 | 0.9998960 |

| accuracy vs official FlashRT output | STATIC | DYNAMIC |
| --- | ---: | ---: |
| max abs | 0.02734375 | 0.017578125 |
| mean abs | 0.00335428 | 0.00264854 |
| p99 abs | 0.01953125 | 0.015625 |
| MSE | 3.004e-05 | 1.599e-05 |
| cosine | 0.9998623 | 0.9999278 |

| component accuracy | STATIC | DYNAMIC |
| --- | ---: | ---: |
| encoder-reference p99 abs | 0.75 | 0.75 |
| encoder-reference cosine | 0.9986721 | 0.9986666 |
| encoder-official p99 abs | 0.875 | 0.78125 |
| encoder-official cosine | 0.9980740 | 0.9984350 |
| vision-projector p99 abs | 0.25 | 0.25 |
| vision-projector cosine | 0.9998775 | 0.9998890 |

| LIBERO Spatial task validation | result |
| --- | ---: |
| original model success rate | 98.2% |
| FP8 with real-data recalibration | keeps original 98.2% level |

### Complete Pipeline, Decoder FFN Dynamic Only

This is a complete `vision -> encoder -> decoder` timing, but only the decoder
FFN scale source differs. Vision and encoder are identical in both runs.

| full e2e, decoder FFN only | STATIC | DYNAMIC | static is |
| --- | ---: | ---: | ---: |
| CUDA graph | 23.33 ms | 27.43 ms | 1.18x faster |
| eager | 26.07 ms | 33.76 ms | 1.30x faster |
| first call | 112.6 ms | 135.4 ms | - |
| action cosine | 0.99997 | 0.99994 | both >=0.9999 |

### Decoder Loop Only

| decoder loop | STATIC | DYNAMIC | static is |
| --- | ---: | ---: | ---: |
| CUDA graph | 10.02 ms | 14.12 ms | 1.41x faster |
| eager | 12.10 ms | 23.79 ms | 1.97x faster |
| action cosine vs ref | 0.99998 | 0.99902 | both >=0.999 |

### Single GeGLU MLP Kernel

| M tokens | STATIC fused | DYNAMIC split | static is |
| --- | ---: | ---: | ---: |
| 50 decode | 0.070 ms | 0.121 ms | 1.73x faster |
| 512 prefill | 0.275 ms | 0.332 ms | 1.21x faster |

BF16 reference timings for the same shapes were 0.151 ms at M=50 and 0.581 ms
at M=512, so static FP8 fused is about 2.1x faster than BF16 on these shapes.

## Files

Reproduction files live under `demos/`:

| file | purpose |
| --- | --- |
| `demos/static-vs-dynamic-fp8/run_static_dynamic_fp8.py` | unified runner for all reproduction commands |
| `demos/static-vs-dynamic-fp8/geglu_static_dynamic_microbench.py` | standalone GeGLU kernel static-vs-dynamic microbench |
| `demos/runtime-demo/dyn_full_all_fp8.py` | strict full all-FP8 e2e, static or dynamic |
| `demos/runtime-demo/dyn_full.py` | full e2e with only decoder FFN dynamic |
| `demos/runtime-demo/sd_decoder.py` | decoder-loop-only static-vs-dynamic benchmark |
| `demos/runtime-demo/pi05_hf_decoder_e2e.py` | PI0.5 HF-kernel e2e runtime and JSON metrics |
| `demos/runtime-demo/pi05_decoder_loop_hub.py` | PI0.5 decoder loop runtime |

Committed data files:

| file | purpose |
| --- | --- |
| `internal-tests/runtime-demo/pi05-real-images-encoder-x-kv-frame50.pt` | real LIBERO normalized image/state/noise/encoder bundle |
| `internal-tests/runtime-demo/pi05-real-encoder-kv-frame50.pt` | decoder-loop bundle with real encoder KV |
| `internal-tests/runtime-demo/pi05-hf-vision-encoder-decoder-frame50-decoder-static-scales.json` | decoder static activation scales |
| `internal-tests/runtime-demo/pi05-hf-vision-encoder-frame50-static-scales.json` | encoder static activation scales |
| `internal-tests/runtime-demo/pi05-decoder-loop-hub-static-scales.json` | decoder-loop static activation scales |

External file needed:

| file | how to provide |
| --- | --- |
| PI0.5 PyTorch checkpoint | set `PI05_CHECKPOINT=/path/to/pi05_libero_pytorch` or pass `--checkpoint /path/to/pi05_libero_pytorch` |

## Environment

Use a Python environment with CUDA, PyTorch, and the Hugging Face `kernels`
package:

```bash
PY=/path/to/python
cd /path/to/FlashRT-HF-kernels
export PI05_CHECKPOINT=/path/to/pi05_libero_pytorch
```

The Hub artifacts are downloaded automatically through `kernels.get_kernel()`.

## Reproduce Everything

```bash
$PY demos/static-vs-dynamic-fp8/run_static_dynamic_fp8.py \
  --checkpoint "$PI05_CHECKPOINT" \
  --out-dir internal-tests/runtime-demo/static-vs-dynamic-fp8 \
  --warmup 8 \
  --iters 30 \
  --microbench-warmup 20 \
  --microbench-iters 100
```

Outputs:

```text
internal-tests/runtime-demo/static-vs-dynamic-fp8/
  all_fp8_e2e_static.log
  all_fp8_e2e_static.json
  all_fp8_e2e_dynamic.log
  all_fp8_e2e_dynamic.json
  decoder_ffn_e2e_static.log
  decoder_ffn_e2e_static.json
  decoder_ffn_e2e_dynamic.log
  decoder_ffn_e2e_dynamic.json
  decoder_loop.log
  geglu_microbench.log
  geglu_microbench.json
  summary.json
```

## Reproduce Individual Rows

Strict full all-FP8 e2e:

```bash
$PY demos/runtime-demo/dyn_full_all_fp8.py static \
  --checkpoint "$PI05_CHECKPOINT" \
  --warmup 8 --iters 30

$PY demos/runtime-demo/dyn_full_all_fp8.py dynamic \
  --checkpoint "$PI05_CHECKPOINT" \
  --warmup 8 --iters 30
```

Complete pipeline, decoder FFN dynamic only:

```bash
$PY demos/runtime-demo/dyn_full.py static \
  --checkpoint "$PI05_CHECKPOINT" \
  --warmup 8 --iters 30

$PY demos/runtime-demo/dyn_full.py dynamic \
  --checkpoint "$PI05_CHECKPOINT" \
  --warmup 8 --iters 30
```

Decoder loop only:

```bash
$PY demos/runtime-demo/sd_decoder.py \
  --checkpoint "$PI05_CHECKPOINT" \
  --warmup 8 --iters 30
```

Standalone GeGLU microbench:

```bash
$PY demos/static-vs-dynamic-fp8/geglu_static_dynamic_microbench.py \
  --shapes headline \
  --warmup 20 \
  --iters 100
```

Short smoke run:

```bash
$PY demos/static-vs-dynamic-fp8/run_static_dynamic_fp8.py \
  --checkpoint "$PI05_CHECKPOINT" \
  --suite microbench \
  --microbench-warmup 2 \
  --microbench-iters 5
```

## Notes

- `dyn_full_all_fp8.py` is the strict comparison: vision, encoder, and decoder
  all use FP8 projection/FFN paths, and the dynamic side recomputes activation
  scales at runtime.
- `dyn_full.py` is kept only to reproduce the older "complete pipeline but
  decoder FFN dynamic only" row.
- `sd_decoder.py` isolates the 10-step decoder loop, where the dynamic split
  overhead is less diluted by one-shot vision/encoder work.
- Dynamic GeGLU cannot use the fused MLP kernel because the hidden activation
  scale is unknown until after the gate/up GEMM and activation. It therefore
  runs as quantize -> FP8 GEMM -> materialized hidden -> amax -> quantize ->
  FP8 GEMM.
