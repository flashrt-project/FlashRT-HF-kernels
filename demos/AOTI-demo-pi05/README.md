# AOTI-demo-pi05 — a pi05-fast optimization stack

A flux-fast-style recipe that makes an **unmodified LeRobot pi05** policy fast
through a toggleable ladder of inference optimizations, benchmarked against the
BF16 eager baseline. Every step is a small, optional change to the stock policy
— a recipe, not a runtime rewrite.

The stack:

```
bf16 eager (baseline)
  + inductor tuning flags
  + FlashRT fused FP8 GeGLU MLP   (action expert + prefix language model)
  + torch.compile  OR  torch.export + AOTInductor
```

## Results (RTX 5090, LIBERO finetune, 10 denoise steps, full hot path)

Full hot path = SigLIP vision (×2 views) + projector + PaliGemma prefix +
10-step denoise + action head. Cosine is the action chunk vs the eager baseline.

| stage | latency | speedup | action cos |
| --- | --- | --- | --- |
| bf16 eager (baseline) | 111.8 ms | 1.00× | 1.0 |
| + torch.compile | 61.7 ms | 1.81× | 1.0 |
| **+ FlashRT FP8 MLP + compile** | **53.0 ms** | **2.11×** | 0.9999 |
| + FP8 MLP + export-aoti | 54.0 ms | 2.07× | 0.9999 |

The bulk of the speedup is `torch.compile` (which LeRobot pi05 also enables by
default); the FlashRT fused FP8 MLP adds ~1.16× on top, losslessly. AOTI matches
warm compile speed — its value is the portable artifact, not extra throughput.

## Run

```bash
pip install kernels
pip install "lerobot[pi,dataset]"
huggingface-cli login        # the PaliGemma tokenizer is gated

python run_benchmark.py                          # the full ladder above
python run_benchmark.py --single --compile-mode export-aoti
python run_benchmark.py --single --no-fp8        # compile-only rung
```

## The optimizations

- **FlashRT fused FP8 GeGLU MLP** — replaces every Gemma MLP (gate/up/down) in
  the action expert and prefix language model with the fused `fp8_geglu_mlp_bf16`
  kernel from `flashrt/flashrt-fp8-swiglu-ffn`, quantizing input with
  `flashrt/flashrt-gemm-epilogues`. Static scales are calibrated on a **real
  observation** (random inputs break pi05's wide-magnitude prefix), in **eager**
  mode (a compiled graph does not fire the calibration hooks). Attention
  projections stay in BF16 — they are small per-token GEMMs where a per-projection
  FP8 swap loses to cuBLAS.
- **torch.compile** — `max-autotune` on the denoise hot path (the runtime layer;
  no manual CUDA graph needed).
- **torch.export + AOTInductor** (`export-aoti`) — compiles the cleanly-exportable
  SigLIP vision embed to a standalone `.pt2` that loads without re-tuning, then
  compiles the denoise loop. AOTI is the route to fast cold starts and ZeroGPU
  Spaces (where each request forks a fresh process and JIT caches do not carry
  over). The 10-step denoise loop does not yet export as one graph (a per-step
  KV-cache `deepcopy`, a dynamic prefix length, and a fake-tensor mask broadcast
  break `torch.export`); compiling the full loop with AOTI is the documented next
  step.
- **Inductor tuning flags** — `coordinate_descent_tuning`,
  `coordinate_descent_check_all_directions`, `epilogue_fusion=False`.

## Files

- `pi05_fast.py` — the optimization library (`optimize()` + the FP8 MLP swap and
  calibration).
- `pi05_aoti.py` — the `torch.export` + AOTInductor route.
- `run_benchmark.py` — the ladder benchmark.

## Notes

- This is the LeRobot-integration recipe. For FlashRT's own hand-built static hot
  path (manual CUDA graph, lower latency), see `demos/runtime-demo`.
- Reaching higher than ~2× losslessly means attacking the 10-step denoise loop
  itself (step caching / fewer steps), which changes the action output and must be
  validated on task success, not cosine alone.
