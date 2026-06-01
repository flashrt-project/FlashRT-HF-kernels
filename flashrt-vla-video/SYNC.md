# Source Sync Plan

Upstream source: `../official/FlashRT`

Candidate source areas:

- `csrc/kernels/patch_embed.*`
- `csrc/kernels/dit_bf16.*`
- `csrc/conv/`
- `csrc/quantize/bf16_*ncdhw*`

## First Source Slice

Start with patch embedding or a BF16 layout/quant helper before moving to
larger video convolution kernels.
