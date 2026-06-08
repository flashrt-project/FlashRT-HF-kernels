# PI0.5 HF Kernel Runtime Demo

This demo tests whether FlashRT Hub kernels can be composed into a clean hot
path without the overhead that appears in simple "replace one PyTorch function"
experiments.

It is not the full FlashRT serving runtime. It is a small HF-compatible runtime
prototype that uses the public `kernels.get_kernel(...)` loading path and then
keeps the hot path clean:

- load Hub kernels once during setup;
- quantize weights once during setup;
- keep static scales as device tensors;
- preallocate all scratch/output buffers;
- call kernels with explicit output tensors;
- optionally capture the fixed-shape path with CUDA Graph.

## Why This Exists

Direct hot replacement is usually too noisy for model-level claims. If every
layer allocates outputs, re-enters Python wrappers, or converts dtype/layout in
the wrong place, microkernel wins can disappear at end-to-end scale.

This demo compares:

| Path | Meaning |
| --- | --- |
| `torch_fp8_reference` | PyTorch FP8 fake-quant reference for correctness and baseline latency. |
| `naive_hub` | Hub kernels called like normal Python functions, allowing wrapper allocations. |
| `runtime_prealloc` | Hub kernels with preloaded ops, static weights/scales, and preallocated buffers. |
| `runtime_cuda_graph` | Same runtime path captured and replayed with CUDA Graph. |

## Run

Use an environment matching the published Hub artifact. On the local validation
machine this is the torch 2.11 CUDA 12.8 smoke environment:

```bash
/home/heima/suliang/PI/.flashrt-hub-smoke-torch211/bin/python \
  demos/pi05-hf-runtime/benchmark_runtime.py \
  --shape pi05_decoder \
  --warmup 10 \
  --iters 50 \
  --cuda-graph
```

Try a larger PI0.5 vision-shaped chain:

```bash
/home/heima/suliang/PI/.flashrt-hub-smoke-torch211/bin/python \
  demos/pi05-hf-runtime/benchmark_runtime.py \
  --shape pi05_vision \
  --warmup 5 \
  --iters 20 \
  --cuda-graph
```

## Interpretation

This is a runtime-overhead benchmark, not a full PI0.5 policy benchmark. The
chain is model-shaped and uses public Hub kernels, but synthetic weights and
inputs keep the test deterministic and independent from private checkpoints.

## RTX 5090 Hub Results

Environment:

- GPU: NVIDIA GeForce RTX 5090
- PyTorch: `2.11.0+cu128`
- Kernel loading path: `get_kernel("flashrt/...", version=1)`
- Correctness reference: PyTorch FP8 fake-quant chain

| Shape | M,K,H,N,layers | Torch FP8 ref us | Naive Hub us | Runtime prealloc us | CUDA Graph us | Runtime vs ref | Graph vs ref | Runtime vs naive | Correctness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `pi05_decoder` | `10,1024,4096,1024,18` | 2573.840 | 406.022 | 397.360 | 339.253 | 6.48x | 7.59x | 1.02x | p99=0.000854, cos=0.999822 |
| `pi05_vision` | `512,1152,4304,1152,27` | 11059.447 | 1819.238 | 1746.138 | 1553.850 | 6.33x | 7.12x | 1.04x | p99=0.000916, cos=0.999726 |
| `groot_vit` | `512,1024,4096,1024,24` | 17025.360 | 2295.232 | 2289.389 | 1870.579 | 7.44x | 9.10x | 1.00x | p99=0.000854, cos=0.999770 |

These rows answer the immediate runtime question:

- The public HF `kernels` loading path is not the fundamental blocker once
  kernels are loaded once and reused.
- Wrapper allocation overhead is measurable but small for these FFN chains
  because the GEMMs dominate.
- CUDA Graph replay is materially useful. It removes much of the Python launch
  overhead around the multi-kernel chain and should be a first-class feature in
  the future FlashRT-HF runtime.
- The larger integration risk is not `get_kernel(...)`; it is replacing too
  small a slice of the model or inserting extra BF16/FP8 conversion islands
  between otherwise fast kernels.

The current public `flashrt-fp8-ffn` package returns BF16 after the down
projection. Therefore this runtime still needs a BF16-to-FP8 quantization step
before each next layer. That is a real integration cost and identifies the
next kernel gap for a stronger HF runtime:

```text
FP8 FFN down projection -> residual/bias -> FP8 quantized output
```

Once that producer kernel exists, adjacent FP8 blocks can pass FP8 activations
directly and the runtime can avoid one extra quantization launch per layer.
