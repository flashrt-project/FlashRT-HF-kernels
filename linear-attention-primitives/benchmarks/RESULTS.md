# Linear Attention Primitives Results

Command:

```bash
python linear-attention-primitives/tests/test_linear_attention_primitives.py --backend source
python linear-attention-primitives/benchmarks/benchmark.py --backend source --warmup 100 --iters 1000
```

Environment:

- GPU: local RTX 5090 validation host
- Backend: source JIT build
- Correctness: 17/17 checks passed
- Reference: PyTorch eager tensor formulas on the same GPU
- Last refreshed: 2026-06-20

## Benchmark

| Workload | Shape | FlashRT us | PyTorch eager us | Speedup |
|---|---:|---:|---:|---:|
| bf16_matvec | N=512,K=128 | 3.805 | 5.744 | 1.51x |
| bf16_matvec | N=1024,K=4096 | 6.148 | 8.202 | 1.33x |
| bf16_matvec | N=10240,K=5120 | 27.527 | 46.086 | 1.67x |
| bf16_smallm_matmul | M=2,N=96,K=5120 | 8.195 | 20.999 | 2.56x |
| bf16_smallm_matmul | M=3,N=96,K=5120 | 8.199 | 20.869 | 2.55x |
| bf16_smallm_matmul | M=4,N=96,K=5120 | 8.199 | 20.811 | 2.54x |
| split_qkv_broadcast | rows=1,heads=16/48,dim=128 | 5.780 | 34.703 | 6.00x |
| split_qkv_broadcast | rows=128,heads=16/48,dim=128 | 6.082 | 40.440 | 6.65x |
| split_qkv_broadcast | rows=1024,heads=16/48,dim=128 | 24.573 | 53.274 | 2.17x |
| partial_rope_qk | rows=1,heads=16,dim=128,rope=64 | 4.488 | 182.071 | 40.57x |
| partial_rope_qk | rows=128,heads=16,dim=128,rope=64 | 4.553 | 184.148 | 40.45x |
| partial_rope_qk | rows=1024,heads=16,dim=128,rope=64 | 12.298 | 185.548 | 15.09x |
| gated_delta_prepare | rows=1,heads=48 | 4.949 | 18.536 | 3.75x |
| gated_delta_prepare | rows=128,heads=48 | 4.944 | 19.116 | 3.87x |
| gated_delta_prepare | rows=1024,heads=48 | 4.928 | 19.166 | 3.89x |

## Shape Contract

- `bf16_matvec`: `N >= 256`, positive-speedup validation shown above.
- `bf16_smallm_matmul`: restricted to tuned AB96 path `2 <= M <= 4`, `N=96`, `K=5120`.
- `split_qkv_broadcast`: rows `1,128,1024`, `q_heads=16`, `kv_heads=16`, `v_heads=48`, `head_dim=128`.
- `partial_rope_qk`: rows `1,128,1024`, `heads=16`, `head_dim=128`, `rope_dim=64`.
- `gated_delta_prepare`: rows `1,128,1024`, `heads=48`.
