# Wan QKV Postprocess Demo

This demo measures a Wan-style transformer block fragment:

1. split packed QKV;
2. RMS-normalize Q and K;
3. apply interleaved RoPE to Q and K.

The fused FlashRT path maps to `flashrt-vla-video::qkv_split_norm_rope_bf16`.
The PyTorch baseline is intentionally eager tensor code because this path is
usually launch-bound and memory-movement-bound in model code.

Wan-relevant shapes:

- Wan2.2 TI2V-5B: `heads=24`, `head_dim=128`, `dim=3072`.
- Wan A14B family: `heads=40`, `head_dim=128`, `dim=5120`.
- Video token lengths: `256,1024,2520,4096`.

## Run

From the repository root:

```bash
python demos/wan-qkv-postprocess/benchmark.py \
  --backend source \
  --mode all \
  --tokens 256,1024,2520,4096 \
  --heads 24,40 \
  --compile-baseline \
  --warmup 20 \
  --iters 100 \
  --output internal-tests/demos/wan-qkv-postprocess/results.json \
  --markdown internal-tests/demos/wan-qkv-postprocess/results.md
```

`--backend source` JIT-compiles the package source against the current PyTorch
environment. Use `--backend installed --artifact <build-variant-dir>` after a
matching HF artifact has been copied locally.

After publishing to the Hub, use the same demo through the public `kernels`
API:

```bash
python demos/wan-qkv-postprocess/benchmark.py \
  --backend hub \
  --repo-id LiangSu8899/flashrt-vla-video \
  --version 1 \
  --mode all \
  --tokens 256,1024,2520,4096 \
  --heads 24,40 \
  --compile-baseline
```

## Interpretation

- `fused_prealloc_us`: model-integration style; caller owns Q/K output buffers.
- `fused_alloc_us`: Python wrapper style; fused kernel allocates Q/K outputs.
- `torch_eager_us`: unfused PyTorch eager split + RMSNorm + RoPE reference.
- `torch.compile us`: compiled PyTorch reference when `--compile-baseline` is
  enabled. The first compilation is excluded from the timed loop.
- `speedup_prealloc`: headline model-block speedup.
- `speedup_alloc`: user-facing wrapper speedup.
- `attention-e2e`: measures `packed_qkv -> Q/K postprocess -> attention
  output`; the same attention implementation is used on both sides, so this
  shows the end-to-end benefit after attention cost is included.
- `self-attention-e2e`: measures `x -> QKV projection -> Q/K postprocess ->
  attention -> output projection`. The FlashRT path packs Wan's separate Q/K/V
  projection weights once and uses one mathematically equivalent packed QKV
  projection before the FlashRT postprocess kernel.

This is not a full Wan generation benchmark. It isolates the reusable kernel
block that can be called from a Wan transformer implementation.

## RTX 5090 Source-Extension Snapshot

Torch `2.9.1+cu128`, RTX 5090, source-extension backend. The table below uses
the long-token retest with `T in {1024,2520,4096}` because those shapes are more
representative for video/VLA workloads than tiny smoke cases.

| Scope | Wan2.2 TI2V-5B vs eager | Wan2.2 TI2V-5B vs compile | Wan A14B vs eager | Wan A14B vs compile | Meaning |
| --- | ---: | ---: | ---: | ---: | --- |
| Q/K postprocess only | 17.12x-33.74x | 4.00x-4.66x | 17.15x-24.32x | 2.23x-5.06x | Direct value of this kernel. |
| Packed-QKV to attention output | 1.96x-2.36x | 1.06x-1.27x | 2.34x-2.83x | 1.09x-1.46x | Attention cost included and held constant. |
| Self-attention sublayer | 1.41x-1.59x | 1.14x-1.35x | 1.25x-1.45x | 1.06x-1.10x | QKV/O projection and attention dominate this wider block. |

Retest result files:

- `internal-tests/demos/wan-qkv-postprocess/results-all-retune-compile-2026-06-03-rtx5090.md`
- `internal-tests/demos/wan-qkv-postprocess/results-all-retune-compile-2026-06-03-rtx5090.json`

The larger FlashRT story should be reported separately from this single-kernel
demo. This demo shows one reusable kernel's contribution. Full FlashRT model
paths can combine multiple math-equivalent kernels across attention, FFN,
epilogues, quantization, residual, layout, and serving glue. Those gains are
orthogonal to sampling-step reduction, distillation, cache reuse, and other
quality/performance trade-off techniques.

`torch.compile` should be reported alongside eager PyTorch for final figures.
The self-attention sublayer table is not the headline for this single kernel:
it intentionally includes QKV projection, attention, and output projection, so
the fused postprocess kernel is only a fraction of the measured runtime.
