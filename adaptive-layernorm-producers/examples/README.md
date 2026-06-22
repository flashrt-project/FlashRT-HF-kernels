# Examples

`basic_usage.py` shows the public Kernel Hub loading path:

```bash
python adaptive-layernorm-producers/examples/basic_usage.py
```

It allocates static output buffers for the FP8 path and uses the package helper
to allocate the NVFP4 packed activation and swizzled scale-factor buffers.
