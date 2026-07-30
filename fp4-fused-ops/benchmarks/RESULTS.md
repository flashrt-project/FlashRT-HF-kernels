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
