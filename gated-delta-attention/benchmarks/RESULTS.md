# Results

Built-artifact benchmark results are pending for v3.

The table below is a local source-extension triage benchmark on
`NVIDIA GeForce RTX 5090`, CUDA `12.8`, PyTorch `2.9.1+cu128`. It compares the
same package's baseline WY CUDA fallback path against the new v3 FLA-style
native CUDA MMA prefill path. It is not a HF Jobs artifact benchmark and should
be regenerated after upload.

Both paths include the same static state reset copy before each timed replay,
so the comparison is apples-to-apples within this package.

| S | Baseline WY fallback us | FLA MMA WY us | Speedup |
| ---: | ---: | ---: | ---: |
| 64 | 2495.942 | 1489.869 | 1.68x |
| 128 | 3559.574 | 1582.773 | 2.25x |
| 512 | 10511.024 | 2768.136 | 3.80x |
| 1024 | 19554.431 | 4174.474 | 4.68x |
