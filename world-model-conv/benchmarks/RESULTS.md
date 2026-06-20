# World Model Conv Results

Command:

```bash
python world-model-conv/tests/test_world_model_conv.py --backend source
python world-model-conv/benchmarks/benchmark.py --backend source --warmup 50 --iters 500
```

Environment:

- GPU: local RTX 5090 validation host
- Backend: source JIT build
- Architecture: `TORCH_CUDA_ARCH_LIST=12.0a`
- Correctness: 3/3 checks passed
- Last refreshed: 2026-06-20

## Benchmark

| Workload | Shape | FlashRT us | PyTorch eager us | Speedup |
|---|---:|---:|---:|---:|
| fp8_conv3d_v18 | N=1,Tc=2,T=4,H=16,W=16,Ci=32,Co=32 | 26.650 | 64.728 | 2.43x |
| fp8_conv3d_v18 | N=2,Tc=2,T=4,H=16,W=24,Ci=64,Co=32 | 49.142 | 72.953 | 1.48x |
| fp8_conv3d_v18 | N=1,Tc=2,T=8,H=32,W=32,Ci=64,Co=64 | 52.922 | 112.643 | 2.13x |
