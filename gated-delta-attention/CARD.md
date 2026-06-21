---
library_name: kernels
tags:
- cuda
- pytorch
- flashrt
- gated-delta
- linear-attention
- qwen3
- transformer
---

# Gated Delta Attention

BF16 Gated DeltaNet recurrent/chunk/WY kernels from FlashRT, packaged for
Hugging Face Kernel Hub. The v3 public profile covers Qwen3.6-style
linear-attention decode recurrence, prefill WY building blocks, and the
native CUDA FLA-style MMA prefill path.

## Available functions

- `gated_delta_recurrent_bf16`
- `gated_delta_recurrent_inout_bf16`
- `gated_delta_recurrent_f32state_bf16io`
- `gated_delta_chunk_bf16`
- `gated_delta_chunk_smem_bf16`
- `lin_split_qkv_broadcast_bf16`
- `lin_split_qkv_gqa_bf16`
- `split_q_gate_bf16`
- `gdn_gating_bf16`
- `gdn_gating_strided_bf16`
- `gdn_chunk_from_conv_smem_bf16`
- `gdn_wy_norm_cumsum_pack_qk_bf16`
- `gdn_wy_kkt_b64_bf16`
- `gdn_wy_solve_tril_b64_f32`
- `gdn_wy_cast_ai_f32_to_bf16`
- `gdn_wy_recompute_wu_b64_bf16`
- `gdn_wy_chunk_h_b64_bf16`
- `gdn_wy_output_o_b64_bf16`
- `gdn_wy_recompute_wu_b64_mma_fla_bf16`
- `gdn_wy_chunk_h_b64_mma_fla_bf16`
- `gdn_wy_output_o_b64_mma_fla_bf16`
- `gdn_wy_output_o_b64_mma_fla_rawk_bf16`

## Usage

```python
from kernels import get_kernel

gdn = get_kernel("flashrt/gated-delta-attention", version=3, trust_remote_code=True)
out = gdn.gated_delta_recurrent_bf16(q, k, v, g, beta, state)
```

The WY helpers use the Qwen3.6 profile: `conv_out=(S,10240)`,
Q/K heads `16`, value heads `48`, head dimension `128`, and 64-token WY
blocks.

The FLA-style path keeps the hot prefill chain in CUDA kernels:

```python
q16_l2, k16_l2, q_pack, _, g_cumsum = gdn.gdn_wy_norm_cumsum_pack_qk_bf16(q16, k16, g)
A = gdn.gdn_wy_kkt_b64_bf16(k16_l2, beta, g_cumsum)
Ai = gdn.gdn_wy_solve_tril_b64_f32(A, S)
Ai_pack = gdn.gdn_wy_cast_ai_f32_to_bf16(Ai, S)
w_pack, u_pack = gdn.gdn_wy_recompute_wu_b64_mma_fla_bf16(k16_l2, v48, beta, g_cumsum, Ai_pack)
h0, _, v_pack, k_pack = gdn.gdn_wy_chunk_h_b64_mma_fla_bf16(k16_l2, w_pack, u_pack, g_cumsum, state)
out = gdn.gdn_wy_output_o_b64_mma_fla_bf16(q_pack, k_pack, v_pack, h0, g_cumsum)
```
