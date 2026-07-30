# Source Sync

Upstream FlashRT source baseline: `official/FlashRT` commit `132049d`.

Copied source:

- `csrc/quantize/fp8_per_token_block_quant.cu`
- `csrc/quantize/fp8_per_token_block_quant.cuh`
- `csrc/kernels/norm_act_to_fp8_block128.cu`

Package-local changes:

- Added a declaration-only header for the norm/activation producers.
- Added checked PyTorch Tensor bindings and fake/meta registrations.
- Replaced the serial GELU/GELU+bias row loop with one CTA per
  `(row, 128-channel block)`. Outputs and scales remain bitwise identical to
  the original FlashRT native kernel, while wide-row latency improves.
- No FlashRT runtime pointer or explicit stream argument is public.

Compile requirements:

- CUDA 12.8 or newer.
- BF16 and FP8 E4M3 support.
- No model-local or FlashRT `third_party` include dependency.
