# Validation

Source validation compares `fp8_conv3d_v18_ncdhw_res_bf16out` against a PyTorch
reference that performs:

1. `torch.cat([cache_x, new_x], dim=1)`,
2. NDHWC to NCDHW conversion,
3. `torch.nn.functional.conv3d` with spatial padding `(0,1,1)`,
4. alpha scale and per-channel bias,
5. BF16 rounding,
6. residual add and final BF16 rounding.

Validation matrix:

- `(N=1,T_cache=2,T=1,H=8,W=8,Ci=32,Co=16)`
- `(N=1,T_cache=2,T=4,H=16,W=16,Ci=32,Co=32)`
- `(N=2,T_cache=2,T=4,H=16,W=24,Ci=64,Co=32)`

Run:

```bash
python tests/test_world_model_conv.py --backend source
python benchmarks/benchmark.py --backend source --warmup 50 --iters 500
kernel-builder check-config .
```

The source JIT validation forces `TORCH_CUDA_ARCH_LIST=12.0a` because the
kernel uses architecture-specific Blackwell FP8 MMA instructions.
