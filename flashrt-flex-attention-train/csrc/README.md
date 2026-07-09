# csrc

CUDA extension placeholder for the FlexAttention replacement package.

The v1 package currently exposes the stable Python Tensor API and SDPA-backed
autograd fallback. Native CUDA forward/backward kernels should be added here
without changing the public functions in `torch-ext/flashrt_flex_attention_train`.
