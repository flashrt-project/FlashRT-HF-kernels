# Source synchronization

The MoE weighted-sum implementation is synchronized from:

- FlashRT revision `cd12ef670bfb0dd8b83ee846be30ace523541313`

- `official/FlashRT/csrc/kernels/moe_weighted_sum_sm120.cu`
- `official/FlashRT/csrc/kernels/moe_weighted_sum_sm120.cuh`

The package binding adds a tensor API and a model-neutral alias for the
existing router. Source kernels remain architecture-specific and unsupported
devices fail explicitly.

## GROOT N1.7 SM110 vector path

- FlashRT revision: `24df793f4fa2d50780aea03b644208c6e0cb4162`
- Copied source: `csrc/kernels/vec_fp16_backbone.cu/.cuh`
- Package dispatch: `csrc/kernels/vec_fp16_dispatch.cu/.cuh`

This additive CUDA 13 SM110 target exposes FP16 RMSNorm/LayerNorm, static FP8
producers, rotate-half RoPE, residual add, and GQA head expansion. Existing
SM120 transformer and MoE APIs are unchanged. All public wrappers use Tensor
arguments and preserve caller-owned in-place/static-buffer semantics where the
native source does.
