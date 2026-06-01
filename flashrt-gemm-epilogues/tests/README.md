# Tests

Current tests cover the first implemented epilogue slice:

- `bias_gelu_quantize_fp8_static_bf16`
- `channel_scale_quantize_fp8_static_bf16`
- `gelu_quantize_fp8_static_bf16`

The tests compare FP8 output against a PyTorch reference expression:

```python
torch.clamp(torch.nn.functional.gelu(x + bias, approximate="tanh") / scale, -448, 448).to(torch.float8_e4m3fn)
```

and the channel-scale expression:

```python
torch.clamp((x * channel_scale) / scale, -448, 448).to(torch.float8_e4m3fn)
```

Package tests must remain Hub-compatible and should not import the adjacent
FlashRT checkout. FlashRT parity tests belong in `../../internal-tests/`.
