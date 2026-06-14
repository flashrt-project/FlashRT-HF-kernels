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
  + fused FP8 QKV                 (pack q/k/v into one FP8 GEMM, + FP8 o_proj)
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
| + sync-free denoise loop | 40.7 ms | 2.75× | 0.9999 |
| **+ fused FP8 QKV (full)** | **37.0 ms** | **3.03×** | 0.9999 |

Latencies are warm steady-state medians (the first call is ~2.5× higher, a
one-time cost). `torch.compile` carries the first 1.81×; the FlashRT fused FP8
kernels add the GeGLU MLP in both Gemma stacks and the GELU MLP in the SigLIP
vision tower; the sync-free pass removes the per-step CPU↔GPU syncs and the
per-step KV-cache copy; fused FP8 QKV packs q/k/v into one GEMM. `export-aoti`
matches warm compile speed — its value is the portable artifact, not extra
throughput.

## Run

```bash
pip install kernels
pip install "lerobot[pi,dataset]"
huggingface-cli login        # the PaliGemma tokenizer is gated

python run_benchmark.py                                                # the full ladder above
python run_benchmark.py --single --sync-free --vision-fp8 --fp8-attn  # full stack, one config
python run_benchmark.py --single --no-fp8                             # compile-only rung
```

The demo loads the published FlashRT Hub packages
`flashrt/flashrt-fp8-swiglu-ffn`, `flashrt/flashrt-fp8-ffn` and
`flashrt/flashrt-gemm-epilogues` via `get_kernel` (see `../../docs/usage.md` for
the package map). The full ladder reloads a compiled policy per rung in one
process; on limited VRAM run rungs individually with `--single` to avoid
accumulating CUDA-graph memory.

## The optimizations

- **FlashRT fused FP8 GeGLU MLP** — replaces every Gemma MLP (gate/up/down) in
  the action expert and prefix language model with the fused `fp8_geglu_mlp_bf16`
  kernel from `flashrt/flashrt-fp8-swiglu-ffn`, quantizing input with
  `flashrt/flashrt-gemm-epilogues`. Static scales are calibrated on a **real
  observation** (random inputs break pi05's wide-magnitude prefix), in **eager**
  mode (a compiled graph does not fire the calibration hooks). The attention
  projections are handled separately: a per-projection FP8 swap of q/k/v loses to
  cuBLAS at these small token counts, so they are only worth FP8 once fused (see
  fused FP8 QKV below).
- **FlashRT fused FP8 GELU MLP** — replaces the SigLIP vision-tower MLP
  (`fc1` → gelu_tanh → `fc2`, with bias) with `fp8_gelu_mlp_bf16` from
  `flashrt/flashrt-fp8-ffn`. The vision tower is large-token, where FP8 pays off;
  it is kept in fp32, so the module casts to BF16 for the FP8 path and back so the
  SigLIP residual is unchanged. Enable with `--vision-fp8`.
- **sync-free denoise loop** (`--sync-free`) — the stock 10-step loop rebuilds a
  tensor from a Python value every step (`torch.tensor(time, device=cuda)` for the
  flow timestep, `torch.tensor(att_masks, device=cuda)` for the suffix mask), ~2
  host↔GPU syncs per step (~21 per call). This precomputes the timestep schedule
  once and builds the suffix mask on-device, dropping syncs to ~2 (bit-exact). It
  also removes the per-step `copy.deepcopy` of the prefix KV cache: the joint
  forward appends the suffix K/V to the shared cache, so the step lets it append
  and then slices each layer back to the prefix length (a traceable tensor op, no
  Python copy). That removes the loop's last graph break — the 10 steps now
  compile as a single graph (a prerequisite for exporting the loop with AOTI;
  full-loop export still needs the dynamic prefix length and mask handled, see
  Notes). It is a runtime monkeypatch of the policy's methods (LeRobot source is
  untouched), so it is slightly more than a module swap; it covers the non-RTC path.
- **fused FP8 QKV** (`--fp8-attn`) — packs the attention `q/k/v` projections into a
  single FP8 GEMM (one quantize, one wider GEMM) instead of three separate ones,
  and optionally runs `o_proj` in FP8. This is the only way FP8 pays off on the
  attention at pi05's small token counts: a per-projection FP8 drop-in actually
  *regresses* (3 tiny GEMMs + 3 quantizes lose to cuBLAS BF16), but the fused
  packed GEMM wins — ~1.5× on the QKV projection at the decode shape, ~−2.5 ms
  end to end. The `o_proj` FP8 part is ~neutral on latency (one small GEMM); it
  just completes FP8 coverage of the attention. It is a runtime monkeypatch of
  `GemmaAttention.forward` (LeRobot source untouched), so like sync-free it is more
  than a module swap. (This mirrors diffusers' `fuse_qkv_projections()` + FP8.)
- **torch.compile** — `max-autotune` on the denoise hot path (the runtime layer;
  no manual CUDA graph needed).
- **torch.export + AOTInductor** (`export-aoti`) — compiles the cleanly-exportable
  SigLIP vision embed to a standalone `.pt2` that loads without re-tuning, then
  compiles the denoise loop. AOTI is the route to fast cold starts and ZeroGPU
  Spaces (where each request forks a fresh process and JIT caches do not carry
  over). The 10-step denoise loop is not AOTI-exported yet: `--sync-free` removes
  the per-step KV-cache copy (so it's a single `torch.compile` graph), but a
  dynamic prefix length and a fake-tensor mask broadcast still break
  `torch.export`; compiling the full loop with AOTI is future work.
- **Inductor tuning flags** — `coordinate_descent_tuning`,
  `coordinate_descent_check_all_directions`, `epilogue_fusion=False`.

## Files

- `pi05_fast.py` — the optimization library: `optimize()` plus the FP8 MLP/vision
  swaps, the sync-free denoise loop, the fused FP8 QKV, and real-observation
  calibration.
- `pi05_aoti.py` — the `torch.export` + AOTInductor route (vision embed).
- `run_benchmark.py` — the ladder benchmark.

## Notes

- This is the LeRobot-integration recipe. For FlashRT's own hand-built static hot
  path (manual CUDA graph, lower latency), see `demos/runtime-demo`.
- After the sync-free pass the whole `sample_actions` compiles as a single graph
  (0 breaks — the KV-cache slice-back replaces the per-step copy). That's a
  prerequisite for AOTI-exporting the loop, though full-loop `torch.export` still
  needs the dynamic prefix length and mask resolved (the `export-aoti` mode
  currently AOTIs only the vision embed). Full-model compile beats
  regional compilation here: ~54 ms (full, `max-autotune`) vs ~66 ms (regional, no
  FP8) — full-model keeps the whole denoise loop under CUDA graphs, while a block
  compiled and called in a Python loop can't stably use CUDA graphs (output
  buffer reuse), so it falls back to no-cudagraphs and pays the launch overhead.
  With `--fp8-attn` the QKV/O projections are FP8 too (fused), so the GEMMs are now
  FP8 across MLP + attention; what's left in BF16 is the attention score/softmax
  itself, which is a small fraction at pi05's short sequences (a flash-attention
  swap measured neutral-to-slower here). Faster than that means attacking the 10
  denoise steps themselves (step caching / fewer steps), which changes the action
  output and must be validated on task success, not cosine.
