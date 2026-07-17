# Examples

Run after installing a compatible Kernel Hub artifact:

```bash
python weight-only-ffn/examples/basic_usage.py
```

The example prepares static W4 and W8 weights once, then invokes complete
SwiGLU FFN regions with reusable buffers through production auto dispatch. Its
`M=1,K=1024,H=4096,N=1024` geometry is inside the measured support domain.
