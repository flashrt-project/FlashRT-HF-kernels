# Source synchronization

The MoE weighted-sum implementation is synchronized from:

- FlashRT revision `cd12ef670bfb0dd8b83ee846be30ace523541313`

- `official/FlashRT/csrc/kernels/moe_weighted_sum_sm120.cu`
- `official/FlashRT/csrc/kernels/moe_weighted_sum_sm120.cuh`

The package binding adds a tensor API and a model-neutral alias for the
existing router. Source kernels remain architecture-specific and unsupported
devices fail explicitly.
