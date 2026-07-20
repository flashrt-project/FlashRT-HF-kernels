# FA2 Seqused Runtime

FlashRT's forward-only FlashAttention-2 runtime surface for static-buffer,
CUDA Graph execution. The package adds device-resident K/V lengths and an
allocation-free Tensor API around FlashRT's vendored FA2 kernels.

This is not a replacement for the general-purpose
`kernels-community/flash-attn2` package. Use the community package for training,
backward, varlen, paged KV cache, or a broad drop-in FlashAttention API. Use this
package when a fixed-shape inference runtime needs to replay one CUDA Graph while
the valid K/V length changes on device.

## Load from Kernel Hub

```python
import torch
from kernels import get_kernel

fa2 = get_kernel("flashrt/fa2-seqused-runtime", version=1)

q = torch.randn(1, 32, 8, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 512, 2, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn_like(k)
seqused_k = torch.tensor([384], device="cuda", dtype=torch.int32)
out, lse = fa2.allocate_outputs(q)

fa2.forward_seqused_static(
    q, k, v, seqused_k, out=out, softmax_lse=lse
)
```

`q`, `k`, and `v` use `(batch, sequence, heads, head_dim)` layout. Static
runtime calls never allocate. `allocate_outputs` and `allocate_workspace` are
setup helpers and must stay outside the captured hot path.

## Public functions

| Function | Purpose |
|---|---|
| `forward` | Convenience forward with output/workspace allocation |
| `forward_static` | Allocation-free fixed-length forward into caller buffers |
| `forward_seqused_static` | Allocation-free BF16 forward with device `seqused_k` |
| `allocate_outputs` | Allocate output and FP32 LSE during setup |
| `allocate_workspace` | Allocate exact split-KV scratch during setup |
| `recommended_num_splits` | Expose the FlashRT split heuristic for planning |

## Contract

- CUDA inference forward only; no backward or dropout.
- Non-causal: FP16/BF16, head dimension 64/96/128/256.
- Causal: BF16, head dimension 128/256.
- For `Sq != Sk`, causal masking uses FlashAttention's bottom-right alignment:
  query row `i` may attend through KV column `i + Sk - Sq`.
- Split-KV: head dimension 96/128/256.
- Query heads must be divisible by KV heads.
- Last dimension must be contiguous. Padded layouts are supported when batch,
  row, and head strides preserve the kernel's 16-byte vector alignment.
- `seqused_k` is contiguous CUDA int32 with one value per batch. Values must be
  in `[1, max_seqlen_k]`.
- SM80 or newer, except SM110/Thor is not claimed by the upstream FlashRT FA2
  runtime at this time.
- Published fat binaries use the lowest target in each supported major family:
  `sm80`, `sm90`, `sm100`, and `sm120`. CUDA cubin compatibility makes `sm80`
  cover the 8.x family (including 8.6 and 8.9) and `sm120` cover the 12.x
  family (including 12.1).

Full FlashRT model runtimes and serving pipelines are maintained at
[LiangSu8899/FlashRT](https://github.com/LiangSu8899/FlashRT).

## Provenance

The FA2 kernel body is derived from FlashAttention-2 under BSD-3-Clause. The
static C runtime and Tensor packaging are from FlashRT. See `SYNC.md` for exact
source revisions and local changes.
