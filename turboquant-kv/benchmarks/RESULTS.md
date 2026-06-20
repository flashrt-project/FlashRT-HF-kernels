# Results

Local source benchmark before unified HF Jobs artifact build.

Environment:

- GPU: NVIDIA GeForce RTX 5090, SM 12.0
- PyTorch: 2.9.1+cu128
- CUDA runtime: 12.8
- Command: `python turboquant-kv/benchmarks/benchmark.py --backend source --warmup 50 --iters 500`
- Last refreshed: 2026-06-20

| Workload | Shape | FlashRT us | PyTorch eager us | Speedup |
|---|---:|---:|---:|---:|
| unpack_packed_bf16 | M=1, D=256, bits=3/4 | 6.089 | 87.479 | 14.37x |
| unpack_packed_mixed | M=1, D=256, bits=3/4 | 6.034 | 86.608 | 14.35x |
| unpack_packed_bf16 | M=4, D=256, bits=3/4 | 6.109 | 89.002 | 14.57x |
| unpack_packed_mixed | M=4, D=256, bits=3/4 | 6.443 | 87.962 | 13.65x |
| unpack_packed_bf16 | M=128, D=256, bits=3/4 | 6.132 | 96.819 | 15.79x |
| unpack_packed_mixed | M=128, D=256, bits=3/4 | 6.071 | 89.580 | 14.76x |
| unpack_packed_bf16 | M=1024, D=256, bits=3/4 | 5.824 | 88.691 | 15.23x |
| unpack_packed_mixed | M=1024, D=256, bits=3/4 | 6.082 | 88.257 | 14.51x |
| unpack_packed_bf16 | M=4096, D=256, bits=3/4 | 6.040 | 86.495 | 14.32x |
| unpack_packed_mixed | M=4096, D=256, bits=3/4 | 6.072 | 88.093 | 14.51x |
| unpack_packed_bf16 | M=32768, D=256, bits=3/4 | 20.041 | 288.160 | 14.38x |
| unpack_packed_mixed | M=32768, D=256, bits=3/4 | 22.551 | 342.855 | 15.20x |
| combine_kv_bf16 | M=1, D=256 | 4.511 | 43.847 | 9.72x |
| combine_kv_bf16 | M=4, D=256 | 4.522 | 43.814 | 9.69x |
| combine_kv_bf16 | M=128, D=256 | 4.796 | 43.681 | 9.11x |
| combine_kv_bf16 | M=1024, D=256 | 4.513 | 43.844 | 9.72x |
| combine_kv_bf16 | M=4096, D=256 | 4.627 | 44.614 | 9.64x |
| combine_kv_bf16 | M=32768, D=256 | 18.430 | 213.252 | 11.57x |

Installed-artifact benchmark must be regenerated after the unified HF Jobs
batch build.
