# Gated Delta Attention

FlashRT native CUDA BF16 Gated DeltaNet / linear-attention state kernels.

The package supports two validated linear-attention producer profiles:

- value/key heads: `48/16` and `32/16`
- key/value head dim: `128`
- BF16 q/k/v/g/beta/out
- BF16 or FP32 recurrent state
- optional in-kernel Q/K L2 normalization
- prefill WY block helpers with 64-token chunks
- FLA-style native CUDA MMA prefill path for WY recompute/chunk/output

## Available Functions

- `gated_delta_recurrent_bf16(q, k, v, g, beta, state, use_qk_l2norm=True, out=None)`
- `gated_delta_recurrent_inout_bf16(q, k, v, g, beta, state_in, use_qk_l2norm=True, state_out=None, out=None)`
- `gated_delta_recurrent_f32state_bf16io(q, k, v, g, beta, state_f32, use_qk_l2norm=True, out=None)`
- `gated_delta_chunk_bf16(q, k, v, g, beta, state, use_qk_l2norm=True, out=None)`
- `gated_delta_chunk_smem_bf16(q, k, v, g, beta, state, use_qk_l2norm=True, out=None)`
- `gated_delta_recurrent_sequence_bf16(q, k, v, g, beta, state, use_qk_l2norm=True, out=None)`
- `lin_split_qkv_broadcast_bf16(conv_out, q48=None, k48=None, v48=None)`
- `lin_split_qkv_broadcast_h_bf16(conv_out, num_v_heads, num_k_heads, head_dim=128, ...)`
- `lin_split_qkv_gqa_bf16(conv_out, q16=None, k16=None, v48=None)`
- `split_q_gate_bf16(q_proj, q_pre=None, gate=None)`
- `gdn_gating_bf16(a, b, neg_exp_A_log, dt_bias, g_out=None, beta_out=None)`
- `gdn_gating_h_bf16(a, b, neg_exp_A_log, dt_bias, num_heads=None, ...)`
- `gdn_gating_strided_bf16(a, b, neg_exp_A_log, dt_bias, rows, a_stride, b_stride, ...)`
- `gdn_gating_strided_h_bf16(a, b, neg_exp_A_log, dt_bias, rows, num_heads, a_stride, b_stride, ...)`
- `gdn_chunk_from_conv_smem_bf16(conv_out, a, b, neg_exp_A_log, dt_bias, state, ...)`
- `gdn_chunk_from_conv_smem_h_bf16(conv_out, a, b, neg_exp_A_log, dt_bias, state, num_v_heads, num_k_heads, ...)`
- `gdn_wy_norm_cumsum_pack_qk_bf16(q16, k16, g, ...)`
- `gdn_wy_kkt_b64_bf16(k16_l2, beta, g_cumsum, A=None)`
- `gdn_wy_solve_tril_b64_f32(A, S, Ai=None)`
- `gdn_wy_cast_ai_f32_to_bf16(Ai, S, Ai_pack=None)`
- `gdn_wy_recompute_wu_b64_bf16(k16_l2, v48, beta, g_cumsum, Ai, ...)`
- `gdn_wy_chunk_h_b64_bf16(k16_l2, u48, w48, g_cumsum, state, ...)`
- `gdn_wy_output_o_b64_bf16(q16_l2, k16_l2, v_new, h0, g_cumsum, out=None)`
- `gdn_wy_recompute_wu_b64_mma_fla_bf16(k16_l2, v48, beta, g_cumsum, Ai_pack, ...)`
- `gdn_wy_chunk_h_b64_mma_fla_bf16(k16_l2, w_pack, u_pack, g_cumsum, state, ...)`
- `gdn_wy_output_o_b64_mma_fla_bf16(q_pack_hv, k_pack_hv, v_pack, h0, g_cumsum, ...)`
- `gdn_wy_output_o_b64_mma_fla_rawk_bf16(q_pack_hv, k16_l2, v_pack, h0, g_cumsum, ...)`
- `gdn_wy_norm_cumsum_pack_qk_h_bf16(q, k, g, num_v_heads, num_k_heads, ...)`
- `gdn_wy_kkt_b64_h_bf16(k_l2, beta, g_cumsum, num_v_heads, num_k_heads, ...)`
- `gdn_wy_solve_tril_b64_h_f32(A, S, num_v_heads, ...)`
- `gdn_wy_cast_ai_h_f32_to_bf16(Ai, S, num_v_heads, ...)`
- `gdn_wy_recompute_wu_b64_mma_fla_h_bf16(...)`
- `gdn_wy_chunk_h_b64_mma_fla_h_bf16(...)`
- `gdn_wy_output_o_b64_mma_fla_h_bf16(...)`
- `gdn_wy_output_o_b64_mma_fla_rawk_h_bf16(...)`

The v5 API covers both decode recurrence and linear-attention prefill/WY
building blocks. It does not package generic FlashAttention.

`gated_delta_recurrent_sequence_bf16` scans `(S,H,128)` in one SM110/SM120
launch. Its recurrent state remains FP32 inside the scan and is converted to
the caller's BF16 `state` tensor only once at the end. This differs from the
legacy chunk contract, which rounds state to BF16 after each token.

The v3 FLA-style MMA path requires NVIDIA `sm_80+` because it uses BF16
`mma.sync` instructions. The package build targets Ampere, Ada, Hopper, and
Blackwell CUDA capabilities, including the native `sm_110a` Thor artifact.

## Usage

```python
from kernels import get_kernel
import torch

gdn = get_kernel("flashrt/gated-delta-attention", version=5, trust_remote_code=True)

B, H, D = 1, 48, 128
q = torch.randn(B, H, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
g = torch.randn(B, H, device="cuda", dtype=torch.bfloat16)
beta = torch.sigmoid(torch.randn(B, H, device="cuda")).to(torch.bfloat16)
state = torch.zeros(B, H, D, D, device="cuda", dtype=torch.bfloat16)

out = gdn.gated_delta_recurrent_bf16(q, k, v, g, beta, state)
```

Generic H32/H16 producer and fused prefill path:

```python
S, Hv, Hk, D = 64, 32, 16, 128
conv_out = torch.randn(S, (2 * Hk + Hv) * D, device="cuda", dtype=torch.bfloat16)
a = torch.randn(S, Hv, device="cuda", dtype=torch.bfloat16)
b = torch.randn_like(a)
neg_exp_A_log = -torch.rand(Hv, device="cuda", dtype=torch.float32)
dt_bias = torch.randn(Hv, device="cuda", dtype=torch.float32)
state = torch.zeros(Hv, D, D, device="cuda", dtype=torch.bfloat16)

q, k, v = gdn.lin_split_qkv_broadcast_h_bf16(conv_out, Hv, Hk)
g, beta = gdn.gdn_gating_h_bf16(a, b, neg_exp_A_log, dt_bias)
out = gdn.gdn_chunk_from_conv_smem_h_bf16(
    conv_out, a, b, neg_exp_A_log, dt_bias, state,
    num_v_heads=Hv, num_k_heads=Hk,
)

# H32/H16 WY prefill uses the same native stages with model-neutral head args.
q16 = conv_out[:, : Hk * D].view(S, Hk, D).contiguous()
k16 = conv_out[:, Hk * D : 2 * Hk * D].view(S, Hk, D).contiguous()
q_l2, k_l2, q_pack, _, g_cumsum = (
    gdn.gdn_wy_norm_cumsum_pack_qk_h_bf16(
        q16, k16, g, num_v_heads=Hv, num_k_heads=Hk
    )
)
A = gdn.gdn_wy_kkt_b64_h_bf16(
    k_l2, beta, g_cumsum, num_v_heads=Hv, num_k_heads=Hk
)
Ai = gdn.gdn_wy_solve_tril_b64_h_f32(A, S, num_v_heads=Hv)
Ai_pack = gdn.gdn_wy_cast_ai_h_f32_to_bf16(Ai, S, num_v_heads=Hv)
w_pack, u_pack = gdn.gdn_wy_recompute_wu_b64_mma_fla_h_bf16(
    k_l2, v, beta, g_cumsum, Ai_pack,
    num_v_heads=Hv, num_k_heads=Hk,
)
h0, _, v_pack, k_pack = gdn.gdn_wy_chunk_h_b64_mma_fla_h_bf16(
    k_l2, w_pack, u_pack, g_cumsum, state,
    num_v_heads=Hv, num_k_heads=Hk,
)
out = gdn.gdn_wy_output_o_b64_mma_fla_h_bf16(
    q_pack, k_pack, v_pack, h0, g_cumsum,
    num_v_heads=Hv, num_k_heads=Hk,
)
```

The generic producer requires `Hv % Hk == 0` and currently supports
`head_dim=128`. `neg_exp_A_log` and `dt_bias` are FP32 by contract; inputs,
state, and outputs are BF16. The fused chunk updates `state` in place and is
CUDA Graph replay safe after normal warmup.

Prefill-style WY pipeline:

```python
S = 128
conv_out = torch.randn(S, 10240, device="cuda", dtype=torch.bfloat16)
a = torch.randn(S, 48, device="cuda", dtype=torch.bfloat16)
b = torch.randn(S, 48, device="cuda", dtype=torch.bfloat16)
neg_exp_A_log = torch.randn(48, device="cuda").float()
dt_bias = torch.randn(48, device="cuda").float()
state = torch.zeros(48, 128, 128, device="cuda", dtype=torch.bfloat16)

q16, k16, v48 = gdn.lin_split_qkv_gqa_bf16(conv_out)
g, beta = gdn.gdn_gating_bf16(a, b, neg_exp_A_log, dt_bias)
q16_l2, k16_l2, _, _, g_cumsum = gdn.gdn_wy_norm_cumsum_pack_qk_bf16(q16, k16, g)
A = gdn.gdn_wy_kkt_b64_mma_bf16(k16_l2, beta, g_cumsum)
Ai = gdn.gdn_wy_solve_tril_b64_f32(A, S)
w48, u48 = gdn.gdn_wy_recompute_wu_b64_bf16(k16_l2, v48, beta, g_cumsum, Ai)
h0, v_new = gdn.gdn_wy_chunk_h_b64_bf16(k16_l2, u48, w48, g_cumsum, state)
out = gdn.gdn_wy_output_o_b64_bf16(q16_l2, k16_l2, v_new, h0, g_cumsum)
```

FLA-style native CUDA MMA prefill path:

```python
S = q16.shape[0]
scale = 128 ** -0.5

q16_l2, k16_l2, q_pack_hv, _, g_cumsum = gdn.gdn_wy_norm_cumsum_pack_qk_bf16(q16, k16, g)
A = gdn.gdn_wy_kkt_b64_bf16(k16_l2, beta, g_cumsum)
Ai = gdn.gdn_wy_solve_tril_b64_f32(A, S)
Ai_pack = gdn.gdn_wy_cast_ai_f32_to_bf16(Ai, S)
w_pack, u_pack = gdn.gdn_wy_recompute_wu_b64_mma_fla_bf16(
    k16_l2, v48, beta, g_cumsum, Ai_pack
)
h0, v_new, v_pack, k_pack_hv = gdn.gdn_wy_chunk_h_b64_mma_fla_bf16(
    k16_l2, w_pack, u_pack, g_cumsum, state
)
out = gdn.gdn_wy_output_o_b64_mma_fla_bf16(
    q_pack_hv, k_pack_hv, v_pack, h0, g_cumsum, scale=scale
)
```

## Validation

See `VALIDATION.md`.
