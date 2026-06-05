# Examples

Run `fp8_gelu_mlp_block.py` after installing `kernels`:

```bash
python flashrt-fp8-ffn/examples/fp8_gelu_mlp_block.py
python flashrt-fp8-ffn/examples/fp8_gelu_mlp_block.py --compile
```

The script loads `flashrt/flashrt-fp8-ffn` from the Hugging Face Kernel Hub and
optionally wraps `fp8_gelu_mlp_bf16` with `torch.compile(fullgraph=True)`.
