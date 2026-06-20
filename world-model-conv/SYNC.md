# Source Sync

Derived from:

- `official/FlashRT/csrc/conv/fp8_conv3d_sm120_v18.cu`

The HF package exposes the tensor API wrapper around the production native CUDA
kernel. Runtime orchestration, quantization staging, and model-specific buffer
management remain in FlashRT.
