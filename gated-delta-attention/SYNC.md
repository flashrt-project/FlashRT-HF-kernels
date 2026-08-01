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
