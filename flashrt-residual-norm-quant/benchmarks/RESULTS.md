# Benchmark Results: flashrt-residual-norm-quant

## RTX 5090 Source-Extension Results

- Device: NVIDIA GeForce RTX 5090
- Backend: local source extension
- Precision gate: FP8 `p99_abs=0` for the initial shape grid.

Commands:

```bash
python flashrt-residual-norm-quant/tests/test_residual_norm_quant.py --backend source --mode full
python flashrt-residual-norm-quant/benchmarks/benchmark.py \
  --backend source \
  --shapes all \
  --warmup 3 \
  --iters 10
```

## Full Shape Sweep

| Shape | Rows,Dim | Kernel | FlashRT us | Eager us | vs eager | Max abs | Mean abs | P99 abs | Cosine | Status |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| pi05_decoder | 10,1024 | rms_norm_quant_fp8_static_bf16 | 4.352 | 46.243 | 10.63x | 0.000000 | 0.000000 | 0.000000 | 1.00000000 | PASS |
| pi05_decoder | 10,1024 | residual_add_rms_norm_quant_fp8_static_bf16 | 3.882 | 67.203 | 17.31x | 0.000000 | 0.000000 | 0.000000 | 1.00000000 | PASS |
| pi05_vision | 512,1152 | rms_norm_quant_fp8_static_bf16 | 4.454 | 43.779 | 9.83x | 0.000000 | 0.000000 | 0.000000 | 1.00000000 | PASS |
| pi05_vision | 512,1152 | residual_add_rms_norm_quant_fp8_static_bf16 | 4.525 | 53.133 | 11.74x | 0.000000 | 0.000000 | 0.000000 | 1.00000012 | PASS |
| groot_vl | 1024,2048 | rms_norm_quant_fp8_static_bf16 | 6.438 | 63.962 | 9.93x | 0.062500 | 0.000000 | 0.000000 | 1.00000012 | PASS |
| groot_vl | 1024,2048 | residual_add_rms_norm_quant_fp8_static_bf16 | 6.573 | 76.202 | 11.59x | 0.000000 | 0.000000 | 0.000000 | 0.99999988 | PASS |
| video_prefill | 2520,2048 | rms_norm_quant_fp8_static_bf16 | 8.714 | 105.354 | 12.09x | 2.000000 | 0.000001 | 0.000000 | 1.00000012 | PASS |
| video_prefill | 2520,2048 | residual_add_rms_norm_quant_fp8_static_bf16 | 10.582 | 139.824 | 13.21x | 0.000000 | 0.000000 | 0.000000 | 1.00000000 | PASS |

## Interpretation

This package is a runtime glue layer. The main value is removing separate
PyTorch residual add, RMSNorm, and FP8 cast operations between adjacent FP8
GEMM/FFN kernels.

The current source-extension rows show roughly `9.8x-17.3x` speedup against
PyTorch eager references on RTX 5090. Built-artifact and multi-hardware rows
are pending.

## Pending

| Stage | Status |
|---|---|
| Kernel-builder artifact build | pending |
| Built-artifact correctness | pending |
| Built-artifact benchmark | pending |
| Multi-hardware matrix | pending |
