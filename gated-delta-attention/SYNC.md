# Source Sync

Synced from FlashRT:

- sequence source revision: `1103633cecfb40ff9159f8fbbebcf972dbf286da`
- SM120 MMA WY KKT revision:
  `e2f4b16cea32bd520c93119b142758693793dfeb`

- `official/FlashRT/csrc/kernels/gated_deltanet_qwen36.cu`
- `official/FlashRT/csrc/kernels/gated_deltanet_qwen36.cuh`
- `official/FlashRT/csrc/kernels/gdn_recurrent_seq_sm120.cu`
- `official/FlashRT/csrc/kernels/gdn_recurrent_seq_sm120.cuh`
- `official/FlashRT/csrc/kernels/gated_delta_wy_kkt_mma.cu`
- `official/FlashRT/csrc/kernels/gated_delta_wy_kkt_mma.cuh`

Public package rename:

- source file name: `gated_delta_attention`
- public API namespace: `flashrt/gated-delta-attention`

The sequence kernel keeps FP32 recurrent state internally and casts it only at
the public BF16 state boundary. The package adds Tensor validation and graph-
safe preallocated output handling; it does not change the CUDA arithmetic.

The v5 generic producer/chunk/WY API parameterizes the existing native CUDA
addressing by `num_v_heads`, `num_k_heads`, and `head_dim`. The original
48/16/128 entry points remain unchanged and dispatch through the same kernel.
The additional validated profile is 32/16/128 with conv width 8192. All WY
workspace and MMA stages use explicit value/key head counts.

The additive `gdn_wy_kkt_b64_mma_bf16` entry retains the established
16-key-head/48-value-head/D128/64-token-chunk Tensor contract and A layout.
Only its K*K^T implementation changes from scalar FP32 chains to BF16-input,
FP32-accumulator WMMA. The public Hub name is model-neutral; the copied native
symbol retains its historical model prefix internally.
