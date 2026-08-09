# fp4-fused-ops Benchmark Results

Source benchmark on NVIDIA GeForce RTX 5090, PyTorch `2.9.1+cu128`.

Command:

```bash
python fp4-fused-ops/benchmarks/benchmark.py \
  --mode headline \
  --warmup 100 \
  --iterations 500 \
  --json-out internal-tests/fp4-fused-ops-source-benchmark-rerun-long.json
```

These kernels are producer/combiner kernels around FP4 GEMM. They are intended
to remove intermediate PyTorch elementwise launches and keep packed FP4/SFA data
on the low-bit path. Rows without a meaningful fused reference report latency
only.

| Shape | Workload | Reference us | FlashRT us | Speedup | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| rows=1, dim=1024 | residual_add_rms_norm_fp4_sfa_v2 | 6.026 | 5.554 | 1.09x | v1 reference |
| rows=1, dim=1024 | silu_mul_fp4_sfa_v2 | 4.095 | 4.105 | 1.00x | v1 reference |
| rows=1, dim=1024 | silu_mul_mul_fp4_sfa_v2 | n/a | 4.106 | n/a | fused AWQ producer latency |
| rows=1, dim=1024 | silu_mul_two_fp4_to_fp4 | n/a | 6.149 | n/a | FP4-to-FP4 combiner latency |
| rows=1, dim=1024 | silu_mul_two_mul_fp4_to_fp4 | n/a | 6.149 | n/a | FP4-to-FP4 AWQ combiner latency |
| rows=10, dim=2048 | residual_add_rms_norm_fp4_sfa_v2 | 6.134 | 5.788 | 1.06x | v1 reference |
| rows=10, dim=2048 | silu_mul_fp4_sfa_v2 | 4.107 | 4.103 | 1.00x | v1 reference |
| rows=10, dim=2048 | silu_mul_mul_fp4_sfa_v2 | n/a | 4.105 | n/a | fused AWQ producer latency |
| rows=10, dim=2048 | silu_mul_two_fp4_to_fp4 | n/a | 6.150 | n/a | FP4-to-FP4 combiner latency |
| rows=10, dim=2048 | silu_mul_two_mul_fp4_to_fp4 | n/a | 6.149 | n/a | FP4-to-FP4 AWQ combiner latency |
| rows=64, dim=2048 | residual_add_rms_norm_fp4_sfa_v2 | 6.142 | 6.125 | 1.00x | v1 reference |
| rows=64, dim=2048 | silu_mul_fp4_sfa_v2 | 4.106 | 4.102 | 1.00x | v1 reference |
| rows=64, dim=2048 | silu_mul_mul_fp4_sfa_v2 | n/a | 4.103 | n/a | fused AWQ producer latency |
| rows=64, dim=2048 | silu_mul_two_fp4_to_fp4 | n/a | 8.443 | n/a | FP4-to-FP4 combiner latency |
| rows=64, dim=2048 | silu_mul_two_mul_fp4_to_fp4 | n/a | 10.471 | n/a | FP4-to-FP4 AWQ combiner latency |
| rows=128, dim=4096 | residual_add_rms_norm_fp4_sfa_v2 | n/a | 7.768 | n/a | v2 only; v1 rejects this dim |
| rows=128, dim=4096 | silu_mul_fp4_sfa_v2 | 6.149 | 4.097 | 1.50x | v1 reference |
| rows=128, dim=4096 | silu_mul_mul_fp4_sfa_v2 | n/a | 4.102 | n/a | fused AWQ producer latency |
| rows=128, dim=4096 | silu_mul_two_fp4_to_fp4 | n/a | 6.149 | n/a | FP4-to-FP4 combiner latency |
| rows=128, dim=4096 | silu_mul_two_mul_fp4_to_fp4 | n/a | 6.405 | n/a | FP4-to-FP4 AWQ combiner latency |

Installed-artifact results should be regenerated after HF Jobs publishes the
Hub package.

## NCDHW and linear-NVFP4 additions

The migrated Tensor APIs were compared directly with raw FlashRT CUDA
launchers. All output buffers, including packed E2M1 values, UE4M3 scale bytes
and causal caches, are bitwise identical. CUDA Graph rows capture 32 launches
and report time per launch.

| Workload | Native us | Wrapper us | Wrapper/native | Graph native us | Graph wrapper us | Graph wrapper/native | Eager us | Compile us | Exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RMSNorm NCDHW `(1,128,5,9,11)` | 4.119 | 4.118 | 1.000 | 2.949 | 2.948 | 1.000 | 38.519 | 29.905 | yes |
| RMSNorm NCDHW `(1,320,3,7,8)` | 8.205 | 8.208 | 1.000 | 5.956 | 5.925 | 0.995 | 37.954 | 42.343 | yes |
| RMSNorm NCDHW `(1,512,1,5,6)` | 12.305 | 12.314 | 1.001 | 10.889 | 10.903 | 1.001 | 38.188 | 41.765 | yes |
| BF16 RMSNorm-SiLU + cache `(1,128,5,9,11)` | 6.152 | 6.160 | 1.001 | 4.549 | 4.550 | 1.000 | 41.962 | 29.894 | yes |
| linear NVFP4 quant `51x1536` | 4.102 | 4.104 | 1.000 | 2.564 | 2.564 | 1.000 | n/a | n/a | yes |
| fused RMS-SiLU-NVFP4 `(1,128,5,9,11)` | 6.163 | 6.169 | 1.001 | 5.614 | 5.601 | 0.998 | n/a | n/a | yes |

PyTorch does not expose a contract-equivalent NVFP4 E2M1 packer with UE4M3
linear scale-byte output, so those rows deliberately do not report a speedup
against the CPU/Python correctness reference.

## BF16 AdaRMS producer twins on Thor

Measured on NVIDIA Thor, PyTorch `2.13.0+cu130`, CUDA 13.0, with 50 warmups
and 500 timed launches. The comparison is against the already qualified FP16
native-family producer with identical launch geometry and SFA output layout;
it is a performance-parity gate, not an eager speedup claim.

| Rows | BF16 AdaRMS us | BF16/FP16 | BF16 gate-res AdaRMS us | BF16/FP16 |
|---:|---:|---:|---:|---:|
| 1 | 10.244 | 0.998 | 10.684 | 0.669 |
| 10 | 5.076 | 0.985 | 10.431 | 0.992 |
| 51 | 5.102 | 0.982 | 10.345 | 0.972 |
| 105 | 6.132 | 0.990 | 10.156 | 0.951 |

The corresponding Thor source gate passed `66/66`. BF16 residual and gate
outputs are exact, rows=10 CUDA Graph replay is bit-identical, and the worst
dequantized NVFP4 cosine over rows `1/10/51/105` is `0.995462`.

## PI0.5 Thor Batch 3 BF16 producers

Source release candidate on NVIDIA Jetson AGX Thor, PyTorch `2.13.0+cu130`,
CUDA 13.0. Timings use caller-owned outputs, CUDA Graph replay, A-B-B-A
ordering, and repeated minimums. Each BF16 Hub producer is compared with the
matching FlashRT native FP16 producer; the LUT combiner comparison invokes the
same native implementation through both bindings.

| Workload | Hub BF16 us | Native FP16 us | Hub/native |
| --- | ---: | ---: | ---: |
| GeGLU -> FP4, rows=10, H=4096 | 5.380 | 7.300 | 0.737x |
| RMSNorm x scale -> FP4, rows=576, D=2048 | 23.099 | 30.726 | 0.752x |
| RMSNorm x scale -> FP4, rows=970, D=2048 | 38.593 | 59.251 | 0.651x |
| LayerNorm -> FP4, rows=512, D=1152 | 26.393 | 27.564 | 0.958x |
| LayerNorm -> FP8, rows=512, D=1152 | 11.714 | 13.414 | 0.873x |
| LayerNorm -> FP4, rows=768, D=1152 | 34.376 | 33.494 | 1.026x |
| LayerNorm -> FP8, rows=768, D=1152 | 19.214 | 19.190 | 1.001x |
| native split-GU LUT combiner, rows=10, H=4096 | 7.333 | 7.429 | 0.987x |

The final strict source gate passed `89/89` without tolerance relaxation. It
includes one-row style broadcast byte parity, in-place residual equality,
BF16 LayerNorm/RMSNorm/GeGLU max/mean/p99/cosine checks, unsupported-shape
rejection, and deterministic CUDA Graph replay. The native LUT combiner's
dequantized result reached cosine `0.995910` against the mathematical
reference.

## BF16 GeGLU producer vectorization on Thor

Pre-publish source candidate on NVIDIA Jetson AGX Thor, PyTorch
`2.13.0+cu130`, CUDA 13.0. The candidate was compared in isolated processes
against the current Hub v1 artifact to avoid extension namespace collisions.
Both implementations produced byte-identical packed E2M1 output and SFA
bytes.

| Workload | Hub v1 us | Source candidate us | Speedup | Packed/SFA exact |
| --- | ---: | ---: | ---: | --- |
| `gelu_mul_nvfp4_bf16`, PI0.5 producer shape | 8.303 | 5.123 | 1.62x | yes |

The candidate replaces scalar BF16 input loads and packed-byte stores with
16-byte vector transactions without changing the public API or quantization
contract. The Thor source gate passed `89/89`; the RTX 5090 source regression
passed `61/61`. Installed-artifact numbers must be regenerated after the Hub
build is published.
