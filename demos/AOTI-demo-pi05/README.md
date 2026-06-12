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
  + FlashRT fused FP8 GELU MLP    (SigLIP vision tower)
  + sync-free denoise loop        (hoist per-step tensor builds out of the loop)
  + torch.compile  OR  torch.export + AOTInductor
```

## Results (RTX 5090, LIBERO finetune, 10 denoise steps, full hot path)

Full hot path = SigLIP vision (×2 views) + projector + PaliGemma prefix +
10-step denoise + action head. Cosine is the action chunk vs the eager baseline.

| stage | latency | speedup | action cos |
| --- | --- | --- | --- |
| bf16 eager (baseline) | 112.3 ms | 1.00× | 1.0 |
| + torch.compile | 61.7 ms | 1.81× | 1.0 |
| + FlashRT FP8 GeGLU MLP | 53.0 ms | 2.11× | 0.9999 |
| + FlashRT FP8 vision MLP | 48.9 ms | 2.28× | 0.9999 |
| **+ sync-free denoise loop (full)** | **40.7 ms** | **2.75×** | 0.9999 |

Latencies are warm steady-state medians (the first call is ~2.5× higher, a
one-time cost). `torch.compile` carries the first 1.81×; the FlashRT fused FP8
kernels add the GeGLU MLP in both Gemma stacks and the GELU MLP in the SigLIP
vision tower; the sync-free pass removes the per-step CPU↔GPU syncs and the
per-step KV-cache copy. `export-aoti` matches warm compile speed — its value is
the portable artifact, not extra throughput.

## Run

```bash
pip install kernels
pip install "lerobot[pi,dataset]"
huggingface-cli login        # the PaliGemma tokenizer is gated

python run_benchmark.py                                    # the full ladder above
python run_benchmark.py --single --sync-free --vision-fp8  # one config only
python run_benchmark.py --single --no-fp8                  # compile-only rung
```

The demo loads the kernels from the Hub via `get_kernel`, so the packages
`flashrt/flashrt-fp8-swiglu-ffn`, `flashrt/flashrt-fp8-ffn` and
`flashrt/flashrt-gemm-epilogues` must be reachable by your account. The full
ladder reloads a compiled policy per rung in one process; on limited VRAM run
rungs individually with `--single` to avoid accumulating CUDA-graph memory.

## The optimizations

- **FlashRT fused FP8 GeGLU MLP** — replaces every Gemma MLP (gate/up/down) in
  the action expert and prefix language model with the fused `fp8_geglu_mlp_bf16`
  kernel from `flashrt/flashrt-fp8-swiglu-ffn`, quantizing input with
  `flashrt/flashrt-gemm-epilogues`. Static scales are calibrated on a **real
  observation** (random inputs break pi05's wide-magnitude prefix), in **eager**
  mode (a compiled graph does not fire the calibration hooks). Attention
  projections stay in BF16 — they are small per-token GEMMs where a per-projection
  FP8 swap loses to cuBLAS.
- **FlashRT fused FP8 GELU MLP** — replaces the SigLIP vision-tower MLP
  (`fc1` → gelu_tanh → `fc2`, with bias) with `fp8_gelu_mlp_bf16` from
  `flashrt/flashrt-fp8-ffn`. The vision tower is large-token, where FP8 pays off;
  it is kept in fp32, so the module casts to BF16 for the FP8 path and back so the
  SigLIP residual is unchanged. Enable with `--vision-fp8`.
- **sync-free denoise loop** (`--sync-free`) — the stock 10-step loop rebuilds a
  tensor from a Python value every step (`torch.tensor(time, device=cuda)` for the
  flow timestep, `torch.tensor(att_masks, device=cuda)` for the suffix mask), ~2
  host↔GPU syncs per step (~21 per call). This precomputes the timestep schedule
  once and caches the constant suffix mask, dropping syncs to ~2 (bit-exact). It
  also replaces the per-step `copy.deepcopy` of the prefix KV cache with a shallow
  copy (new cache + layer objects, shared tensors), avoiding a full KV-tensor copy
  each step — correctness-equivalent because the joint forward's `cache.update()`
  reassigns via `cat`. It is a runtime monkeypatch of the policy's methods
  (LeRobot source is untouched), so it is slightly more than a module swap; it
  covers the non-RTC path.
- **torch.compile** — `max-autotune` on the denoise hot path (the runtime layer;
  no manual CUDA graph needed).
- **torch.export + AOTInductor** (`export-aoti`) — compiles the cleanly-exportable
  SigLIP vision embed to a standalone `.pt2` that loads without re-tuning, then
  compiles the denoise loop. AOTI is the route to fast cold starts and ZeroGPU
  Spaces (where each request forks a fresh process and JIT caches do not carry
  over). The 10-step denoise loop does not yet export as one graph (a per-step
  KV-cache `deepcopy`, a dynamic prefix length, and a fake-tensor mask broadcast
  break `torch.export`); compiling the full loop with AOTI is future work.
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
- Graph structure is no longer the lever: after the sync-free pass the whole
  `sample_actions` compiles with a single graph break, and forcing a full CUDA
  graph (`mode="reduce-overhead"`) is *slower* than `max-autotune` at pi05's small
  shapes (autotuned kernels beat pure graph replay). So `max-autotune` is the
  compiled ceiling for this FP8 coverage. The remaining headroom is compute —
  attention and the QKV/O projections still run in BF16 — which needs FP8 with a
  fused norm→FP8 producer for the projections (more than a drop-in, it edits the
  attention path). A fully traceable KV-cache path is still what would unblock
  full-loop AOTI export for ZeroGPU Spaces, separate from latency. Faster than
  that means attacking the 10 steps themselves (step caching / fewer steps), which
  changes the action output and must be validated on task success, not cosine.
