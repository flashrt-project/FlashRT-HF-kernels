# flashrt/sageattention3-blackwell

Speed-first FP4 self-attention for SM120a Blackwell GPUs, packaged from the
Apache-2.0 SageAttention3 implementation in `thu-ml/SageAttention`.

Use this package as an explicitly gated lower-precision tier for long video or
audio sequences. Its preferred `sage3_prefill_fp4_bf16` entry accepts raw BF16
NHD Q/K/V and owns Sage3 centering, padding, FP4 quantization, and correction
preparation behind a caller-owned CUDA Graph-safe workspace. Head dimensions
64 and 128 are supported. It does not support GQA/MQA in v1 and does not replace
the higher-fidelity
`flashrt/sageattention2-blackwell` default.

See `README.md` for preprocessing, API usage, accuracy expectations, and the
complete capability contract.
