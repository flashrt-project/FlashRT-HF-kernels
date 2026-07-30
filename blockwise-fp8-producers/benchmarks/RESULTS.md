# Benchmark Results

Local source-extension validation on NVIDIA GeForce RTX 5090, PyTorch 2.9.1,
CUDA 12.8. Timings use 50 warmup and 500 measured iterations for native and
wrapper calls; eager/compile use 20 warmup and 100 measured iterations.

Correctness:

- source matrix: `102/102` passed;
- original FlashRT native output vs Tensor wrapper: FP8 codes and scales
  bitwise equal for every parity row;
- mathematical FP8 reference: worst cosine `0.99963`, worst max/peak ratio
  below `0.036`, worst mean/mean ratio below `0.024`;
- raw width 4304 and non-contiguous inputs: rejected.

Representative native parity and strong baselines:

| Op | Shape | Original native us | Tensor wrapper us | Wrapper/native | Eager us | Compile us | Bitwise |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| quantize | `1x4096` | 2.019 | 2.414 | 1.195 | 30.335 | 23.713 | yes |
| layer norm + quant | `1x4096` | 14.335 | 14.340 | 1.000 | 40.654 | 31.891 | yes |
| RMSNorm + quant | `1x4096` | 10.241 | 10.245 | 1.000 | 55.284 | 30.268 | yes |
| GELU+bias + quant | `1x4096` | 14.333 | 2.345 | 0.164 | 38.846 | 24.825 | yes |
| quantize | `51x4096` | 2.762 | 2.594 | 0.939 | 29.790 | 24.026 | yes |
| layer norm + quant | `51x4096` | 14.336 | 14.337 | 1.000 | 41.139 | 32.121 | yes |
| RMSNorm + quant | `51x4096` | 10.240 | 10.247 | 1.001 | 55.688 | 30.649 | yes |
| GELU+bias + quant | `51x4096` | 14.334 | 2.805 | 0.196 | 39.230 | 24.753 | yes |
| quantize | `277x9216` | 12.290 | 12.290 | 1.000 | 38.971 | 24.605 | yes |
| layer norm + quant | `277x9216` | 28.690 | 28.688 | 1.000 | 59.699 | 38.453 | yes |
| RMSNorm + quant | `277x9216` | 20.478 | 20.483 | 1.000 | 75.874 | 35.019 | yes |
| GELU+bias + quant | `277x9216` | 28.686 | 12.291 | 0.428 | 54.939 | 24.421 | yes |
| quantize | `1024x1152` | 6.150 | 6.150 | 1.000 | 29.607 | 23.634 | yes |
| layer norm + quant | `1024x1152` | 6.152 | 6.159 | 1.001 | 40.908 | 33.670 | yes |
| RMSNorm + quant | `1024x1152` | 6.148 | 6.153 | 1.001 | 55.405 | 30.970 | yes |
| GELU+bias + quant | `1024x1152` | 6.148 | 6.153 | 1.001 | 39.059 | 24.515 | yes |

The original-native column uses the unmodified FlashRT pointer/stream API.
The Tensor wrapper column uses the Hub-style public API with preallocated
outputs. The package-local parallel GELU path is faster than original native
for wide rows and remains bitwise identical.

The isolated M=1 quantizer has about 0.4 us of Python/Torch dispatcher overhead.
Static runtime users should place it inside a compiled/captured region; it is
not used as a standalone Python call in the intended hot pipeline.
