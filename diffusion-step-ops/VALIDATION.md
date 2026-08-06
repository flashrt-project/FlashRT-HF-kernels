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
- Action-tail boundaries `(32,7)`, `(257,51)` and `(4096,1024)`.
- Bias/tail matrices `(1,16,7)`, `(51,64,32)` and `(105,257,256)`.
- Cosmos3-Edge production contracts:
  - flat velocity/action tail `(1,201,920, 3,840)`;
  - bias and zero tail `(60,64,9)`;
  - exact direct output and bitwise CUDA Graph replay.
- UniPC orders `(0,1)`, `(1,1)`, `(1,2)` and `(2,2)`.

Correctness gates:

- exact BF16/FP16 rounded match against PyTorch reference for covered ops.
- exact FP32 match for BF16-to-FP32 cast and decode postprocess.
- fullgraph compile parity for action-tail and UniPC APIs.
- raw Cosmos Edge native parity and direct/CUDA Graph performance comparison.

Run:

```bash
python diffusion-step-ops/tests/test_diffusion_step_ops.py --backend source
```
