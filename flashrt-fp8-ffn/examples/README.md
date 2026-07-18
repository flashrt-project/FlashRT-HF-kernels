# Examples

Run `fp8_gelu_mlp_block.py` after installing `kernels`:

```bash
python flashrt-fp8-ffn/examples/fp8_gelu_mlp_block.py
python flashrt-fp8-ffn/examples/fp8_gelu_mlp_block.py --compile
python flashrt-fp8-ffn/examples/fp8_linear_bias.py
python flashrt-fp8-ffn/examples/fp8_linear_bias.py --compile
```

The script loads `flashrt/flashrt-fp8-ffn` from the Hugging Face Kernel Hub and
optionally wraps `fp8_gelu_mlp_bf16` with `torch.compile(fullgraph=True)`.
`fp8_linear_bias.py` demonstrates the BF16 region API with preallocated
scratch, direct Hub loading, compilation, and CUDA Graph-friendly buffers.
