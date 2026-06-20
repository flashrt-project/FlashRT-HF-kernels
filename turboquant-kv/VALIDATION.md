# Validation

Source correctness covers:

- unpack BF16 and mixed BF16/FP32 outputs
- bit-width combinations:
  - K/V bits `(3, 4)`
  - `(2, 3)`
  - `(3, 3)`
  - `(4, 4)`
- M values:
  - `1`
  - `4`
  - `128`
  - `1024`
- combine M values:
  - `1`
  - `4`
  - `128`
  - `1024`
  - `4096`

Correctness reference:

- packed bytes are decoded directly in PyTorch
- QJL bits are mapped to `{-1, +1}`
- combine reference is:
  `K = k_norm * (k_mse + coef * k_rnorm * k_qjl)`,
  `V = v_norm * v_unit`

Run:

```bash
python turboquant-kv/tests/test_turboquant_kv.py --backend source
```
