# Validation

Source correctness covers:

- BF16 add/euler/CFG/cast for shapes:
  - `(1024,)`
  - `(1025,)`
  - `(4, 4096)`
  - `(2, 16, 32, 64)`
- FP16 CFG on the same shapes.
- Video ops for:
  - `(1, 4, 5, 16, 16)`
  - `(2, 8, 9, 8, 8)`
  - `(1, 16, 17, 16, 24)`

Correctness gates:

- exact BF16/FP16 rounded match against PyTorch reference for covered ops.
- exact FP32 match for BF16-to-FP32 cast and decode postprocess.

Run:

```bash
python diffusion-step-ops/tests/test_diffusion_step_ops.py --backend source
```
