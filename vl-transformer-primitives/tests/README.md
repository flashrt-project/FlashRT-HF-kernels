# Tests

`test_vl_transformer_primitives.py` validates source and installed artifacts against
PyTorch references for:

- Qwen3 decode Q RMSNorm + full RoPE staging.
- Qwen3 decode K RMSNorm + full RoPE + V cache write.
- Device-position KV writes used by CUDA Graph decode loops.
- Qwen3-VL/Cosmos3 style BF16 vision token average pooling.

Run source tests:

```bash
python vl-transformer-primitives/tests/test_vl_transformer_primitives.py --backend source
```
