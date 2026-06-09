# FlashRT HF Runtime Demo

This directory is the first model-runtime prototype built on top of public
FlashRT Hugging Face Kernel Hub packages.

It is intentionally separate from the package-level demos:

- package demos show one kernel family in isolation;
- this runtime demo composes multiple Hub packages into one fixed-shape hot
  path;
- the hot path loads kernels once, owns persistent buffers, avoids timed-loop
  allocation, and supports CUDA Graph replay.

This is not the upstream FlashRT serving runtime. The current Hub path is a
checkpoint-backed PI0.5 runtime bridge for validating whether Hugging Face
Kernel Hub packages can drive a clean model pipeline without losing
microkernel wins to Python/wrapper overhead.

The current real-input milestone runs the model hot path from normalized
LIBERO frame/prompt/state/noise bundle through HF-kernel SigLIP vision,
projector, Gemma encoder, PI0.5 decoder, 10-step denoise, and action output.
The timed hot path has `torch_gaps=[]` and supports CUDA Graph replay. Input
preprocessing and reference-bundle capture still use the official OpenPI /
FlashRT stack, so this should be described as a HF-kernel runtime E2E, not as
a complete OpenPI policy wrapper replacement.

## What It Covers

The default `pi05_hotpath` profile is a PI0.5 decoder-shaped static hot path:
`D=1024`, `heads=8`, `head_dim=128`, `action_rows=10`, `FFN=4096`.

The larger `vla_video_hotpath` profile keeps the original VLA/video stress
shape: `D=3072`, `heads=24`, `head_dim=128`, `FFN=8192`.

Both profiles exercise:

- world-model/video layout glue from `flashrt/flashrt-spatiotemporal-layout`;
- joint V/A/U packed-QKV postprocess from `flashrt/flashrt-qkv-cache-rope`;
- decode Q staging and device-position KV cache write from
  `flashrt/flashrt-qkv-cache-rope`;
- PI0.5-style decoder packed GQA QKV split, adjacent-pair RoPE, and K/V cache
  write from a rebuilt `flashrt-qkv-cache-rope` artifact;
- V/A/U residual-gate updates from `flashrt/flashrt-vla-residual-gates`;
- adaptive norm plus FP8 activation producer from
  `flashrt/flashrt-adaptive-norms`;
- static FP8 GeGLU/SwiGLU FFN stack from `flashrt/flashrt-fp8-swiglu-ffn`.

## Run

Use a Python environment matching one of the published Hub variants. The local
validation environment is:

```bash
/home/heima/suliang/PI/.flashrt-hub-smoke-torch211/bin/python \
  demos/runtime-demo/pi05_runtime_demo.py \
  --profile pi05_hotpath \
  --layers 4 \
  --ffn-activation gelu \
  --attention-backend sdpa \
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

Before the rebuilt GeGLU artifact is uploaded to the Hub, validate the local
`build/<variant>` directory explicitly:

```bash
/home/heima/suliang/PI/.flashrt-hub-smoke-torch211/bin/python \
  demos/runtime-demo/pi05_runtime_demo.py \
  --profile pi05_hotpath \
  --layers 4 \
  --ffn-activation gelu \
  --local-ffn-artifact flashrt-fp8-swiglu-ffn/build/torch211-cxx11-cu128-x86_64-linux \
  --attention-backend sdpa \
  --warmup 5 \
  --iters 20 \
  --cuda-graph
```

`--local-ffn-artifact` only overrides `flashrt-fp8-swiglu-ffn`; all other
packages still load through `kernels.get_kernel("flashrt/...", version=1)`.

To validate the rebuilt QKV package before upload, add the local qkv artifact
and switch the decoder QKV path to the PI0.5 GQA cache API:

```bash
/home/heima/suliang/PI/.flashrt-hub-smoke-torch211/bin/python \
  demos/runtime-demo/pi05_runtime_demo.py \
  --profile pi05_hotpath \
  --layers 1 \
  --ffn-activation silu \
  --attention-backend sdpa \
  --decoder-qkv-backend gqa-cache \
  --local-qkv-artifact flashrt-qkv-cache-rope/build/torch211-cxx11-cu128-x86_64-linux \
  --warmup 2 \
  --iters 5 \
  --cuda-graph
```

On 2026-06-09 this PI0.5-shaped smoke run passed on RTX 5090 with
`graph_us=94.637`, `p99_abs=0.066406`, and `cosine=0.999069`.

For an end-to-end staging report that keeps the baselines separate:

```bash
python demos/runtime-demo/pi05_e2e_runner.py \
  --openpi-baseline-mode docker \
  --container pi0-stablehlo-test \
  --container-repo /workspace/PI/FlashRT-HF-kernels \
  --container-python /workspace/PI/FlashRT-HF-kernels/internal-tests/envs/openpi-baseline/bin/python \
  --container-openpi-root /workspace/PI/openpi_src/src \
  --container-checkpoint /workspace/PI/checkpoints/pi05_libero_pytorch \
  --cuda-graph \
  --output internal-tests/runtime-demo/pi05-e2e-staging.json
```

That runner records three independent rows:

- `openpi_pytorch_baseline`: official PyTorch/OpenPI baseline using real
  `model.safetensors` weights. Use `--openpi-baseline-mode docker` for the
  local `pi0-stablehlo-test` container. The default `probe` mode only checks
  dependency readiness.
- `flashrt_full_fp16_sanity`: checkpoint-backed FlashRT PI0.5 full-FP16 sanity
  run. This proves the local checkpoint/runtime path works, but it is not the
  public PyTorch baseline.
- `hf_kernel_hub_runtime_hotpath`: public HF Kernel Hub hot path using
  persistent buffers and optional CUDA Graph replay.

To run only the official OpenPI/PyTorch baseline inside the local container:

```bash
docker exec pi0-stablehlo-test bash -lc '
cd /workspace/PI/FlashRT-HF-kernels &&
PYTHONPATH=/workspace/PI/openpi_src/src \
python3 demos/runtime-demo/pi05_openpi_baseline.py \
  --openpi-root /workspace/PI/openpi_src/src \
  --checkpoint /workspace/PI/checkpoints/pi05_libero_pytorch \
  --num-views 2 \
  --steps 10 \
  --warmup 5 \
  --iters 20 \
  --compile off \
  --output internal-tests/runtime-demo/pi05-openpi-baseline.json
'
```

The OpenPI baseline should run from an isolated venv, not from the container's
global Python. The local venv is intentionally under ignored `internal-tests/`:

```bash
docker exec pi0-stablehlo-test bash -lc '
set -euo pipefail
VENV=/workspace/PI/FlashRT-HF-kernels/internal-tests/envs/openpi-baseline
python3 -m venv --system-site-packages "$VENV"
"$VENV/bin/python" -m pip install "transformers==4.53.2"
"$VENV/bin/python" - <<PY
import pathlib, shutil, transformers
src = pathlib.Path("/workspace/PI/openpi_src/src/openpi/models_pytorch/transformers_replace")
dst = pathlib.Path(transformers.__file__).resolve().parent
for item in src.iterdir():
    target = dst / item.name
    if item.is_dir():
        shutil.copytree(item, target, dirs_exist_ok=True)
    else:
        shutil.copy2(item, target)
from transformers.models.siglip import check
assert check.check_whether_transformers_replace_is_installed_correctly()
print("openpi baseline venv ready", transformers.__version__, dst)
PY
'
```

The script verifies `transformers==4.53.2` and OpenPI's
`transformers_replace` before constructing the model. The container's global
Python can stay on its default `transformers` version.

To validate a checkpoint-backed Hub-kernel island with real PI0.5 weights and
static activation scales, first capture real OpenPI decoder activations:

```bash
docker exec pi0-stablehlo-test bash -lc '
cd /workspace/PI/FlashRT-HF-kernels &&
PYTHONPATH=/workspace/PI/openpi_src/src \
/workspace/PI/FlashRT-HF-kernels/internal-tests/envs/openpi-baseline/bin/python \
  demos/runtime-demo/pi05_capture_openpi_ffn_activations.py \
  --openpi-root /workspace/PI/openpi_src/src \
  --checkpoint /workspace/PI/checkpoints/pi05_libero_pytorch \
  --family decoder \
  --layer 0 \
  --num-views 2 \
  --steps 10 \
  --output internal-tests/runtime-demo/pi05-openpi-decoder0-mlp-acts.pt
'
```

Then run the Hub-kernel FFN island in the HF kernel environment:

```bash
/home/heima/suliang/PI/.flashrt-hub-smoke-torch211/bin/python \
  demos/runtime-demo/pi05_real_weight_swiglu.py \
  --checkpoint /home/heima/suliang/PI/checkpoints/pi05_libero_pytorch \
  --family decoder \
  --layer 0 \
  --rows 100 \
  --activation-file internal-tests/runtime-demo/pi05-openpi-decoder0-mlp-acts.pt \
  --warmup 10 \
  --iters 50 \
  --output internal-tests/runtime-demo/pi05-realact-swiglu-decoder0.json
```

That script loads real safetensors weights, applies the same per-tensor FP8
weight quantization contract used by FlashRT, calibrates static activation
scales, and runs `flashrt/flashrt-fp8-swiglu-ffn`. It is an FFN island test,
not full policy E2E.

To stage the checkpoint-backed PI0.5 decoder denoise loop with real decoder
weights, official FlashRT-style time/style precompute, the rebuilt QKV cache
kernel, FP8 GeGLU, SDPA attention, and CUDA Graph:

```bash
/home/heima/suliang/PI/.flashrt-hub-smoke-torch211/bin/python \
  demos/runtime-demo/pi05_decoder_loop_hub.py \
  --checkpoint /home/heima/suliang/PI/checkpoints/pi05_libero_pytorch \
  --layers 18 \
  --steps 10 \
  --local-qkv-artifact flashrt-qkv-cache-rope/build/torch211-cxx11-cu128-x86_64-linux \
  --local-ffn-artifact flashrt-fp8-swiglu-ffn/build/torch211-cxx11-cu128-x86_64-linux \
  --warmup 1 \
  --iters 2 \
  --cuda-graph \
  --output internal-tests/runtime-demo/pi05-decoder-loop-hub-18l-smoke.json
```

On 2026-06-09 this 18-layer, 10-step decoder-loop staging run passed on
RTX 5090 with `graph_us=13612.816`, `p99_abs=0.140625`, and
`cosine=0.959428` against the matching quantized PyTorch reference.

This is materially closer to full PI0.5 than the synthetic hot path, but it is
still not full policy E2E. The remaining torch gaps are BF16
action/QKV/O/final projection GEMMs, the SDPA GQA attention backend, and simple
gated residual adds. Those are the next kernels or package surfaces needed for
a fully public pure-kernel decoder pipeline.

Current real-activation island coverage is an island-level probe, not a
publishable full-policy result. OpenPI PI0.5/Gemma uses GeGLU
(`gelu_pytorch_tanh(gate) * up`), so full-model replacement must call
`fp8_geglu_mlp_bf16`, not the package's SwiGLU/SiLU path.

For a strict bridge experiment that runs the full OpenPI `sample_actions`
path and replaces selected Gemma MLP layers:

```bash
docker exec pi0-stablehlo-test bash -lc '
cd /workspace/PI/FlashRT-HF-kernels &&
PYTHONPATH=/workspace/PI/openpi_src/src \
/workspace/PI/FlashRT-HF-kernels/internal-tests/envs/openpi-baseline/bin/python \
  demos/runtime-demo/pi05_openpi_hub_ffn_e2e.py \
  --openpi-root /workspace/PI/openpi_src/src \
  --checkpoint /workspace/PI/checkpoints/pi05_libero_pytorch \
  --num-views 2 \
  --steps 10 \
  --warmup 1 \
  --iters 3 \
  --kernel-source direct \
  --replace-families encoder \
  --replace-layers 0 \
  --output internal-tests/runtime-demo/pi05-openpi-direct-geglu-e2e-encoder0.json
'
```

The GeGLU kernel implementation itself is staged-correct: for synthetic shapes
including the PI0.5 encoder FFN shape `(M=560, K=2048, H=16384, N=2048)`,
`fp8_geglu_mlp_bf16` matches `fp8_gemm_bf16 -> gelu_mul_quant ->
fp8_gemm_bf16` with max error `0` and cosine `1.0`.

The naive OpenPI monkeypatch path is not promoted as a demo. It adds
Python/tensor wrapper dispatch inside the model loop and uses local one-shot
FP8 scales rather than the production FlashRT calibration/runtime contract. A
publishable PI0.5 HF runtime must reuse FlashRT-style static scales,
persistent buffers, and CUDA Graph capture across the whole hot path.

## Interpretation

Use this benchmark to answer runtime engineering questions:

- whether `kernels.get_kernel(...)` can be kept out of the hot path;
- whether the public wrappers can run allocation-free when outputs are passed;
- how much CUDA Graph replay helps a composed multi-kernel path;
- which remaining model islands still need new FlashRT kernels before a true
  full PI0.5 demo is meaningful.

Do not compare the synthetic hot path against the upstream FlashRT serving
runtime. For ecosystem claims, use the official OpenPI/PyTorch model path as
the denominator and use checkpoint-backed Hub-kernel islands or full
checkpoint-backed runtime rows as the numerator.

The staging runner exists to prevent accidental baseline mixing. Environment
smoke rows are intentionally omitted from this document and must not be used as
public baselines.

## RTX 5090 Current Staging Results

Real-input PI0.5 HF-kernel runtime path on RTX 5090:

- Input source: real LIBERO rollout frame 50.
- Boundary: official OpenPI / FlashRT stack exports normalized images,
  prompt/lang embedding, state, initial noise, calibration, and reference
  tensors.
- Timed path: normalized LIBERO images -> HF-kernel SigLIP vision/projector ->
  HF-kernel Gemma encoder KV -> HF-kernel PI0.5 decoder -> 10-step denoise ->
  action.
- Hot-path gaps: `torch_gaps=[]`.
- CUDA Graph: `ok`.
- Static calibration: computed run and loaded-calibration run both pass.

| Path | Scope | Latency ms | Hz | Notes |
| --- | --- | ---: | ---: | --- |
| OpenPI/PyTorch BF16 | First model call after construction | 257.078 | 3.89 | Real checkpoint, official OpenPI model path. This is the current conservative baseline until a full real-LIBERO policy-wrapper benchmark is added. |
| FlashRT HF kernels | Real-input model runtime hot path, CUDA Graph replay | 22.464 | 44.52 | Normalized LIBERO bundle through HF kernels, persistent buffers, static calibration, CUDA Graph replay. |

The HF-kernel graph path is `11.44x` faster than the current OpenPI/PyTorch
first-call baseline. A future policy-wrapper benchmark should report input
preprocessing and bundle capture separately from the model runtime hot path.

Correctness for
`internal-tests/runtime-demo/pi05-hf-vision-encoder-decoder-frame50-static-loaded.json`:

| Comparison | Max abs | Mean abs | P99 abs | Cosine |
| --- | ---: | ---: | ---: | ---: |
| Action vs HF reference | 0.011719 | 0.001716 | 0.007812 | 0.999965 |
| Action vs official FlashRT decoder output | 0.015625 | 0.002160 | 0.011719 | 0.999947 |

Kernel coverage for the timed path:

```text
patch_im2col_bf16
layer_norm_bf16
bf16_linear_bf16
bf16_linear_bias_bf16
qkv_split_bf16
flash_attn2.fwd
bf16_gemm_bias_gelu
bias_residual_bf16
rms_norm_bf16
rms_norm_quant_fp8_static_bf16
qkv_split_rope_kvcache_bf16
fp8_geglu_mlp_bf16
gate_residual_bf16
ada_rms_norm_style_bf16
CUDA Graph replay
```

HF Kernel Hub runtime prototype, run from the published Hub packages:

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
| `pi05_hotpath` | 4 | 2431.566 | 271.728 | 227.166 | 243.386 | 8.95x | 10.70x | p99=0.000135, rms=0.000193 |
| `vla_video_hotpath` | 4 | 10701.723 | 1013.077 | 975.704 | 989.922 | 10.56x | 10.97x | p99=0.000123, rms=0.000181 |

PR619 builder-pin local artifact rerun with the rebuilt GeGLU FFN package:

| Profile | Layers | FFN | Attention | PyTorch eager us | Hub runtime us | CUDA Graph us | Graph + input copy us | Runtime vs eager | Graph vs eager | Correctness |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `pi05_hotpath` | 4 | GeGLU | none | 2444.662 | 284.568 | 226.616 | 251.198 | 8.59x | 10.79x | p99=0.000418, rms=0.000642 |
| `vla_video_hotpath` | 4 | GeGLU | none | 10784.192 | 1016.798 | 999.323 | 992.525 | 10.61x | 10.79x | p99=0.000351, rms=0.000595 |
| `small` | 2 | GeGLU | SDPA | 944.307 | 95.702 | 42.416 | 43.696 | 9.87x | 22.26x | p99=0.066406, cos=0.995436 |
| `pi05_hotpath` | 4 | GeGLU | SDPA | 2339.429 | 297.115 | 248.650 | 252.787 | 7.87x | 9.41x | p99=0.000446, rms=0.000634 |
| `vla_video_hotpath` | 4 | GeGLU | SDPA | 10699.641 | 1054.454 | 1022.397 | 1038.592 | 10.15x | 10.47x | p99=0.000381, rms=0.000596 |

`Attention=none` keeps the previous package-composition hot path for launch
overhead studies. `Attention=SDPA` runs a real non-causal attention over the
Q/K/V tensors produced by `flashrt/flashrt-qkv-cache-rope`; this removes the
old QKV liveness placeholder while keeping the attention backend separate from
FlashRT's production FA2 backend. A future pass can swap SDPA for the
published `kernels-community/flash-attn2` package once its Hub variant is
available in the validation environment.

Latest Hub staging rerun on the same RTX 5090:

- `hf_kernel_hub_runtime_hotpath`: `pi05_hotpath`, 4 layers, CUDA Graph
  `227.166 us`, graph plus input copy `243.386 us`, around `10.7x` vs
  synthetic PyTorch eager for the PI0.5 decoder-shaped profile.
- `hf_kernel_hub_runtime_hotpath`: `vla_video_hotpath`, 4 layers, CUDA Graph
  `975.704 us`, graph plus input copy `989.922 us`, around `11.0x` vs
  synthetic PyTorch eager for the larger VLA/video-shaped profile.
- `pi05_real_weight_swiglu_ffn`: decoder layer 0, rows 10, real checkpoint
  weights plus static activation scales, p50 `19.232 us` vs BF16 PyTorch
  reference p50 `87.296 us` (`4.52x`). Correctness vs quantized reference:
  max abs `0.00635`, p99 abs `0.00208`, cosine `0.99993`.
- `pi05_real_weight_swiglu_ffn` with captured OpenPI activations: decoder
  layer 0, rows 100, p50 `28.000 us` vs BF16 PyTorch reference p50
  `124.448 us` (`4.61x`). Correctness vs quantized reference: cosine
  `0.99998`; max abs is larger (`1.0`) because this compares a cuBLASLt FP8
  accumulation path against a PyTorch dequantized reference and should be read
  as an island profiling check, not a bit-exact production acceptance test.

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
- PI0.5/Gemma uses GeGLU (`gelu_pytorch_tanh(gate) * up`). The current public
  Hub artifact row above uses `--ffn-activation silu` only because the
  published `flashrt-fp8-swiglu-ffn` artifact predates the GeGLU op. Rebuild
  and publish that package before using `--ffn-activation gelu` as the PI0.5
  row.
