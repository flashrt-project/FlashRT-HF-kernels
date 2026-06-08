# FlashRT HF Runtime Demo

This directory is the first model-runtime prototype built on top of public
FlashRT Hugging Face Kernel Hub packages.

It is intentionally separate from the package-level demos:

- package demos show one kernel family in isolation;
- this runtime demo composes multiple Hub packages into one fixed-shape hot
  path;
- the hot path loads kernels once, owns persistent buffers, avoids timed-loop
  allocation, and supports CUDA Graph replay.

This is not the full upstream FlashRT serving runtime and it is not yet a
checkpoint-backed PI0.5 policy benchmark. It is a PI0.5-shaped runtime
simulator for validating whether the HF Kernel Hub path can support a clean
model pipeline without losing microkernel wins to Python/wrapper overhead.

## What It Covers

The default `pi05_hotpath` profile exercises:

- world-model/video layout glue from `flashrt/flashrt-spatiotemporal-layout`;
- joint V/A/U packed-QKV postprocess from `flashrt/flashrt-qkv-cache-rope`;
- decode Q staging and device-position KV cache write from
  `flashrt/flashrt-qkv-cache-rope`;
- V/A/U residual-gate updates from `flashrt/flashrt-vla-residual-gates`;
- adaptive norm plus FP8 activation producer from
  `flashrt/flashrt-adaptive-norms`;
- static FP8 SwiGLU FFN stack from `flashrt/flashrt-fp8-swiglu-ffn`.

## Run

Use a Python environment matching one of the published Hub variants. The local
validation environment is:

```bash
/home/heima/suliang/PI/.flashrt-hub-smoke-torch211/bin/python \
  demos/runtime-demo/pi05_runtime_demo.py \
  --profile pi05_hotpath \
  --layers 4 \
  --warmup 10 \
  --iters 50 \
  --cuda-graph \
  --markdown internal-tests/runtime-demo/pi05-hotpath-rtx5090.md \
  --output internal-tests/runtime-demo/pi05-hotpath-rtx5090.json
```

For a quicker smoke run:

```bash
/home/heima/suliang/PI/.flashrt-hub-smoke-torch211/bin/python \
  demos/runtime-demo/pi05_runtime_demo.py \
  --profile small \
  --layers 2 \
  --warmup 3 \
  --iters 10 \
  --cuda-graph
```

## Interpretation

Use this benchmark to answer runtime engineering questions:

- whether `kernels.get_kernel(...)` can be kept out of the hot path;
- whether the public wrappers can run allocation-free when outputs are passed;
- how much CUDA Graph replay helps a composed multi-kernel path;
- which remaining model islands still need new FlashRT kernels before a true
  full PI0.5 demo is meaningful.

Do not compare this against the upstream FlashRT serving runtime. For ecosystem
claims, compare against the official PyTorch/eager model path once checkpoint
loading and static calibration are wired in.

## RTX 5090 Initial Results

Environment:

- GPU: NVIDIA GeForce RTX 5090
- PyTorch: `2.11.0+cu128`
- CUDA runtime: `12.8`
- Kernel loading path: public `kernels.get_kernel("flashrt/...", version=1)`
- Timed runtime path: preloaded Hub ops, persistent buffers, explicit output
  tensors, optional CUDA Graph replay

| Profile | Layers | PyTorch eager us | Hub runtime us | CUDA Graph us | Graph + input copy us | Runtime vs eager | Graph vs eager | Correctness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `small` | 2 | 1067.482 | 73.357 | 31.194 | 33.382 | 14.55x | 34.22x | p99=0.040039, cos=0.997217 |
| `pi05_hotpath` | 4 | 10520.880 | 988.763 | 974.947 | 991.435 | 10.64x | 10.79x | p99=0.000123, rms=0.000181 |

Reading these rows:

- For small multi-kernel chains, CUDA Graph removes a large fraction of launch
  overhead.
- For the PI0.5-shaped profile, large FP8 GEMMs dominate, so graph replay only
  slightly improves latency. The main win is composing the Hub kernels as a
  continuous preallocated runtime rather than calling tiny replacements from
  Python.
- `Graph + input copy` includes a representative static-input refresh before
  replay. This is the number to use when estimating real runtime integration
  cost.
- The `pi05_hotpath` output is near zero for this random synthetic seed, so
  cosine is intentionally not used as a hard gate there; p99/max error remain
  tiny. Package-level correctness is still covered by each kernel package's
  full installed-artifact tests.
