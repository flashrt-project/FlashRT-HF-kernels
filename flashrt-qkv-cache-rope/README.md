# flashrt-qkv-cache-rope

Tensor-facing FlashRT QKV split, Q/K RMSNorm, and RoPE kernels for Hugging Face
`kernels`.

This package targets VLA, VLM, and video-model attention staging where the
attention core itself can be provided by FA2/FA3/SDPA, but the QKV postprocess
chain is still expensive in PyTorch:

```text
packed QKV -> split Q/K -> RMSNorm Q/K -> RoPE Q/K -> attention-ready Q/K
decode Q/K/V -> RMSNorm Q/K -> rotate-half RoPE Q/K -> Q stage / KV cache write
```

## Exported APIs

- `qkv_split_norm_rope_bf16(packed_qkv, norm_q_weight, norm_k_weight, freqs_re, freqs_im, heads, head_dim, rope_seq_len=None, eps=1e-6, q_out=None, k_out=None)`
- `qkv_split_bias_norm_rope_v_bf16(packed_qkv, qkv_bias, norm_q_weight, norm_k_weight, freqs_re, freqs_im, heads, head_dim, rope_seq_len=None, eps=1e-6, q_out=None, k_out=None, v_out=None)`
- `qkv_split_bias_norm_rope_v_cat_bf16(packed_qkv, qkv_bias, norm_q_weight, norm_k_weight, freqs_re, freqs_im, heads, head_dim, video_offset, q_cat_out, k_cat_out, v_cat_out, rope_seq_len=None, eps=1e-6)`
- `qkv_split_joint3_cat_bf16(packed_v, qkv_v_bias, norm_v_q_weight, norm_v_k_weight, freqs_re, freqs_im, packed_a, norm_a_q_weight, norm_a_k_weight, packed_u, norm_u_q_weight, norm_u_k_weight, heads, head_dim, q_cat_out, k_cat_out, v_cat_out, rope_seq_len=None, eps_v=1e-6, eps_a=1e-6, eps_u=1e-6)`
- `decode_q_norm_rope_stage_bf16(q_pre, q_norm_weight, cos, sin, eps=1e-6, q_out=None)`
- `decode_k_norm_rope_kvwrite_bf16(k_pre, v_pre, k_norm_weight, cos, sin, eps=1e-6, k_cache_dst=None, v_cache_dst=None)`
- `decode_k_norm_rope_kvwrite_devpos_bf16(k_pre, v_pre, k_norm_weight, cos, sin, cur_pos, k_cache, v_cache, eps=1e-6)`

Tensor conventions:

- `packed_qkv`: BF16, shape `(batch, seq_len, 3 * heads * head_dim)`
- `norm_q_weight`: BF16, shape `(heads * head_dim,)`
- `norm_k_weight`: BF16, shape `(heads * head_dim,)`
- `freqs_re`, `freqs_im`: FP32, shape `(>= rope_seq_len, head_dim / 2)`
- non-cat outputs: BF16, shape `(batch, seq_len, heads, head_dim)`
- cat outputs: BF16, shape `(batch, total_seq_len, heads, head_dim)`
- decode `q_pre`, `k_pre`, `v_pre`: BF16, shape `(heads, 128)`
- decode `cos`, `sin`: BF16, shape `(64,)`, rotate-half RoPE contract
- decode cache direct outputs: BF16, shape `(n_kv_heads, 128)`
- decode devpos cache tensors: BF16, shape `(max_seq_len, n_kv_heads, 128)`;
  `cur_pos` is a CUDA int32 scalar or one-element tensor.

The package includes both standalone Q/K(/V) output APIs and direct joint
workspace APIs. The `joint3` API is the VLA-oriented path: it fuses video QKV
bias + Q/K RMSNorm + RoPE, action Q/K RMSNorm, und Q/K RMSNorm, and direct
Q/K/V writes into a single attention-ready workspace.

The decode APIs are single-token hot-path helpers. `decode_q_norm_rope_stage`
writes Q staging buffers. The KV-write APIs write K/V directly into either a
caller-provided cache slot or a slot selected by a device-side `cur_pos`, which
is useful for CUDA Graph replay.

## Minimal Usage

```python
from kernels import get_kernel
import torch

ops = get_kernel("flashrt/flashrt-qkv-cache-rope", version=1, trust_remote_code=True)

packed_qkv = torch.randn((1, 1024, 3 * 24 * 128), device="cuda", dtype=torch.bfloat16)
q_w = torch.ones((24 * 128,), device="cuda", dtype=torch.bfloat16)
k_w = torch.ones((24 * 128,), device="cuda", dtype=torch.bfloat16)
theta = torch.randn((1024, 64), device="cuda")
freqs_re = torch.cos(theta).contiguous()
freqs_im = torch.sin(theta).contiguous()

q, k = ops.qkv_split_norm_rope_bf16(
    packed_qkv,
    q_w,
    k_w,
    freqs_re,
    freqs_im,
    heads=24,
    head_dim=128,
)
```

VLA joint workspace path:

```python
q_cat, k_cat, v_cat = ops.qkv_split_joint3_cat_bf16(
    packed_v,
    qkv_v_bias,
    norm_v_q_weight,
    norm_v_k_weight,
    freqs_re,
    freqs_im,
    packed_a,
    norm_a_q_weight,
    norm_a_k_weight,
    packed_u,
    norm_u_q_weight,
    norm_u_k_weight,
    heads=24,
    head_dim=128,
    q_cat_out=q_cat,
    k_cat_out=k_cat,
    v_cat_out=v_cat,
)
```

Decode cache-write path:

```python
q_pre = torch.randn((24, 128), device="cuda", dtype=torch.bfloat16)
k_pre = torch.randn((8, 128), device="cuda", dtype=torch.bfloat16)
v_pre = torch.randn((8, 128), device="cuda", dtype=torch.bfloat16)
q_w = torch.ones((128,), device="cuda", dtype=torch.bfloat16)
k_w = torch.ones((128,), device="cuda", dtype=torch.bfloat16)
cos = torch.ones((64,), device="cuda", dtype=torch.bfloat16)
sin = torch.zeros((64,), device="cuda", dtype=torch.bfloat16)

q_buf = ops.decode_q_norm_rope_stage_bf16(q_pre, q_w, cos, sin)

k_cache = torch.empty((4096, 8, 128), device="cuda", dtype=torch.bfloat16)
v_cache = torch.empty_like(k_cache)
cur_pos = torch.tensor([17], device="cuda", dtype=torch.int32)
ops.decode_k_norm_rope_kvwrite_devpos_bf16(
    k_pre,
    v_pre,
    k_w,
    cos,
    sin,
    cur_pos,
    k_cache,
    v_cache,
)
```

## Validation

```bash
python flashrt-qkv-cache-rope/tests/test_qkv_cache_rope.py --backend source --mode full
python flashrt-qkv-cache-rope/benchmarks/benchmark.py --backend source --shapes all
```

Current RTX 5090 source-extension rows pass across no-bias, bias+V,
bias+cat-workspace, joint3 workspace, decode Q stage, direct KV-write, and
device-position KV-write paths. The source benchmark includes single-stream
Wan/VL rows, VLA-shaped `joint3_vla`, and single-token decode rows. Current
source rows show roughly `24x-38x` for single-stream QKV postprocess,
`36x-86x` for joint workspace rows, and `29x-30x` for decode Q/KV-write rows
against PyTorch eager references on RTX 5090. Built-artifact and
multi-hardware validation are pending for this expanded API surface.
