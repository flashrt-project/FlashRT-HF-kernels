# Grouped MoE GEMM

FlashRT grouped NVFP4 (W4A4) GEMM kernels for expert-sorted Blackwell prefill
tiles. The API consumes packed E2M1 values and CUTLASS-compatible swizzled
UE4M3 scale-factor buffers and emits BF16.

```python
from kernels import get_kernel
k = get_kernel("flashrt/grouped-moe-gemm", version=1)
y = k.grouped_nvfp4_gemm_bf16(
    a_packed, weights_packed, a_scales, weight_scales,
    expert_alpha, tile_expert, tile_rows=64,
)
```

See [CARD.md](CARD.md) before integrating: packed formats and expert tiling are
part of the ABI. Complete serving pipelines are maintained in
https://github.com/flashrt-project/FlashRT.
