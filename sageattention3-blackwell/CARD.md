# flashrt/sageattention3-blackwell

Speed-first FP4 self-attention for SM120a Blackwell GPUs, packaged from the
Apache-2.0 SageAttention3 implementation in `thu-ml/SageAttention`.

Use this package as an explicitly gated lower-precision tier for long video or
audio sequences. It supports contiguous NHD FP16/BF16 input, head dimensions
64 and 128, caller-owned workspaces, and CUDA Graph replay. It does not support
GQA/MQA in v1 and does not replace the higher-fidelity
`flashrt/sageattention2-blackwell` default.

See `README.md` for preprocessing, API usage, accuracy expectations, and the
complete capability contract.
