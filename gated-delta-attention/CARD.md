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
Hugging Face Kernel Hub. The v5 API extends the model-neutral parameterized
`Hv/Hk` producer and fused recurrent prefill path. Validated profiles include
`Hv/Hk/D=48/16/128` and `32/16/128`.

## Available functions

- `gated_delta_recurrent_bf16`
- `gated_delta_recurrent_inout_bf16`
- `gated_delta_recurrent_f32state_bf16io`
- `gated_delta_chunk_bf16`
- `gated_delta_chunk_smem_bf16`
- `lin_split_qkv_broadcast_bf16`
- `lin_split_qkv_broadcast_h_bf16`
- `lin_split_qkv_gqa_bf16`
- `split_q_gate_bf16`
- `gdn_gating_bf16`
- `gdn_gating_h_bf16`
- `gdn_gating_strided_bf16`
- `gdn_gating_strided_h_bf16`
- `gdn_chunk_from_conv_smem_bf16`
- `gdn_chunk_from_conv_smem_h_bf16`
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
- `gdn_wy_norm_cumsum_pack_qk_h_bf16`
- `gdn_wy_kkt_b64_h_bf16`
- `gdn_wy_solve_tril_b64_h_f32`
- `gdn_wy_cast_ai_h_f32_to_bf16`
- `gdn_wy_recompute_wu_b64_h_bf16`
- `gdn_wy_chunk_h_b64_h_bf16`
- `gdn_wy_output_o_b64_h_bf16`
- `gdn_wy_recompute_wu_b64_mma_fla_h_bf16`
- `gdn_wy_chunk_h_b64_mma_fla_h_bf16`
- `gdn_wy_output_o_b64_mma_fla_h_bf16`
- `gdn_wy_output_o_b64_mma_fla_rawk_h_bf16`

## Usage

```python
from kernels import get_kernel

gdn = get_kernel("flashrt/gated-delta-attention", version=5, trust_remote_code=True)
out = gdn.gated_delta_recurrent_bf16(q, k, v, g, beta, state)
```

The WY helpers use the Qwen3.6 profile: `conv_out=(S,10240)`,
Q/K heads `16`, value heads `48`, head dimension `128`, and 64-token WY
blocks.

The generic H32 producer profile uses `conv_out=(S,8192)`, Q/K heads `16`,
value heads `32`, and head dimension `128`. Q/K are broadcast `16 -> 32`.
Per-head `neg_exp_A_log` and `dt_bias` must remain FP32. Version 5 also
supports the complete 64-token WY prefill chain for this H32/H16 profile;
all `_h_bf16` functions take explicit `num_v_heads` and `num_k_heads`.

```python
S, Hv, Hk, D = 64, 32, 16, 128
q, k, v = gdn.lin_split_qkv_broadcast_h_bf16(conv_out, Hv, Hk)
g, beta = gdn.gdn_gating_h_bf16(a, b, neg_exp_A_log, dt_bias)
out = gdn.gdn_chunk_from_conv_smem_h_bf16(
    conv_out, a, b, neg_exp_A_log, dt_bias, state,
    num_v_heads=Hv, num_k_heads=Hk,
)
```

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
