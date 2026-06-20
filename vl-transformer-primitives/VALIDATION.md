# Validation

Current source validation covers:

- Q decode RMSNorm + RoPE staging for `n_q_heads` in `1, 4, 8, 16, 32, 40`.
- K decode RMSNorm + RoPE and V cache copy for the same head counts.
- Device-position KV cache writes for CUDA Graph decode loops.
- Vision token pooling for Qwen3-VL/Cosmos3-style shapes:
  - `nv=1, h=w=16, dim=1024, pool=2`
  - `nv=2, h=w=16, dim=1152, pool=2`
  - `nv=4, h=w=16, dim=2048, pool=4`
  - `nv=2, h=w=32, dim=1024, pool=4`

Correctness gates:

- max absolute error against PyTorch BF16 reference <= `0.015625`
- cosine similarity >= `0.9999`
- V cache copy exact

Run:

```bash
python vl-transformer-primitives/tests/test_vl_transformer_primitives.py --backend source
```

Installed artifact validation uses the same script with `--backend installed`.
