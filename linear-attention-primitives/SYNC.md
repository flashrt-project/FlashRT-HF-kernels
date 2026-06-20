# Source Sync

This package is derived from production FlashRT kernels in:

- `official/FlashRT/csrc/kernels/bf16_matvec_qwen36.cu`
- `official/FlashRT/csrc/kernels/bf16_matmul_qwen36.cu`
- `official/FlashRT/csrc/kernels/qwen36_misc.cu`
- `official/FlashRT/csrc/kernels/gated_deltanet_qwen36.cu`

Only the model-agnostic tensor helper subset is exported here. Runtime-specific
pointer APIs, speculative decoding orchestration, and larger DeltaNet state
scan kernels remain in the FlashRT runtime repository.
