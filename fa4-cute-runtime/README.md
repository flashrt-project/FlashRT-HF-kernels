# fa4-cute-runtime

Forward-only, SM100-family FlashAttention-4 CuTe DSL package used by FlashRT's
production GROOT N1.7 and PI0.5 Thor runtimes. In addition to the standard
D48/D72/D128 path, it includes the dedicated D256 2CTA forward kernel needed
by PI0.5's PaliGemma encoder.

The package exposes `flash_attn_func`, `flash_attn_varlen_func`, and the
preallocated `forward_static` CUDA Graph entry, including `seqused_k` for
fixed-shape padded K/V. It keeps the vendored
implementation in the private `flashrt_fa4` namespace and vendors the required
quack helper subset, so it cannot replace or mutate upstream `flash_attn`
imports.

See [CARD.md](CARD.md) for usage and [VALIDATION.md](VALIDATION.md) for the
release gate.
