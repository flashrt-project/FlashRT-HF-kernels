# Source Sync

Synced from FlashRT:

- sequence source revision: `1103633cecfb40ff9159f8fbbebcf972dbf286da`

- `official/FlashRT/csrc/kernels/gated_deltanet_qwen36.cu`
- `official/FlashRT/csrc/kernels/gated_deltanet_qwen36.cuh`
- `official/FlashRT/csrc/kernels/gdn_recurrent_seq_sm120.cu`
- `official/FlashRT/csrc/kernels/gdn_recurrent_seq_sm120.cuh`

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
