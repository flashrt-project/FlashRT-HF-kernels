# Examples

Run the minimal FP8 SwiGLU MLP block example after installing `kernels` or
after adding a built artifact to `PYTHONPATH`:

```bash
python flashrt-fp8-swiglu-ffn/examples/fp8_swiglu_mlp_block.py
python flashrt-fp8-swiglu-ffn/examples/fp8_swiglu_mlp_block.py --compile
```

The example loads `flashrt/flashrt-fp8-swiglu-ffn`, allocates FP8 input and
weights, and calls `fp8_swiglu_mlp_bf16`.
