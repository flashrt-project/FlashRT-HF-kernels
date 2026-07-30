# Results

Source correctness on RTX 5090: 29/29 checks passed.

Benchmark environment:

- NVIDIA GeForce RTX 5090
- source extension, current CUDA stream
- preallocated outputs
- 100 warmup and 500 measured iterations
- equivalent eager and `torch.compile(fullgraph=True)` regions

| Op | Shape | Native us | Wrapper us | Eager us | Compile us | Wrapper/native |
|---|---:|---:|---:|---:|---:|---:|
| adaptive | B1 S1 D1280 P16 | 4.104 | 4.108 | 70.329 | 30.354 | 1.001 |
| adaptive | B1 S40 D1536 P64 | 4.105 | 4.106 | 70.897 | 36.711 | 1.000 |
| adaptive | B1 S51 D2048 P64 | 6.145 | 6.151 | 71.194 | 37.214 | 1.001 |
| adaptive | B1 S105 D1280 P128 | 4.103 | 4.102 | 71.317 | 37.385 | 1.000 |
| adaptive | B2 S277 D2048 P320 | 6.154 | 6.150 | 71.719 | 38.627 | 0.999 |
| swiglu | S1 D1280 P16 | 4.041 | 4.061 | 35.492 | 23.131 | 1.005 |
| swiglu | S40 D1536 P64 | 4.096 | 4.101 | 35.531 | 29.407 | 1.001 |
| swiglu | S51 D2048 P64 | 4.099 | 4.101 | 35.959 | 29.752 | 1.000 |
| swiglu | S105 D1280 P128 | 4.100 | 4.089 | 35.631 | 29.738 | 0.997 |
| swiglu | S554 D2048 P640 | 4.104 | 4.104 | 38.946 | 29.855 | 1.000 |

The native column invokes the CUDA launcher through a raw-pointer binding.
The wrapper column invokes the public Tensor API with preallocated outputs.
The benchmark asserts bitwise-identical raw/wrapper FP8 output before timing.
The eager and compile columns implement the complete equivalent producer,
including tail zero-fill; they are not CPU-loop references.

Installed-artifact and Hub performance are pending the unified release build.
