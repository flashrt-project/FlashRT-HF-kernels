# Static vs Dynamic FP8 Demo

This directory contains the reproduction entry points for the PI0.5 static-vs-
dynamic FP8 comparison.

Detailed commands, expected outputs, and measured numbers are in:

```text
docs/static-vs-dynamic-fp8.md
```

Quick smoke:

```bash
cd /path/to/FlashRT-HF-kernels
PY=/path/to/python
$PY demos/static-vs-dynamic-fp8/run_static_dynamic_fp8.py \
  --suite microbench \
  --microbench-warmup 2 \
  --microbench-iters 5
```

Full reproduction:

```bash
export PI05_CHECKPOINT=/path/to/pi05_libero_pytorch
$PY demos/static-vs-dynamic-fp8/run_static_dynamic_fp8.py \
  --checkpoint "$PI05_CHECKPOINT" \
  --warmup 8 \
  --iters 30
```
