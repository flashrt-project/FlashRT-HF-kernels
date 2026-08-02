# Source Sync

The architecture targets intentionally use different native sources behind one
stable Tensor API package.

SM120 sources are derived from the production FlashRT world-model convolution
family, including:

- `csrc/conv/fp8_conv3d_sm120_v18.cu`
- `csrc/conv/fp8_causal_conv3d_sm120.cu`
- `csrc/conv/fp8_conv2d_3x3_sm120.cu`
- `csrc/conv/nvfp4_causal_conv3d*_sm120.cu`

The SM110 BF16 source is copied byte-for-byte from FlashRT commit
`9972f0f4e31a12d69848c007fbdeed6081d62c87`:

- `csrc/conv/bf16_conv3d_v0_sm110.cu`

Package-local architecture stubs keep the common binding linkable without
compiling the other architecture's implementation. They are weak symbols
because kernel-builder may combine both eligible targets in a CUDA 13 variant;
the corresponding real implementation then wins at link time. The SM110
operation is an explicit experimental API: it passes correctness on Thor but
loses to cuDNN on the measured Cosmos3-Edge sites, so it is never selected by
default.

Runtime orchestration, quantization staging, model-specific buffer management,
and automatic backend policy remain outside this package.
