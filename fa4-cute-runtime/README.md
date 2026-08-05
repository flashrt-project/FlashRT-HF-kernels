# fa4-cute-runtime

Forward-only, SM100-family FlashAttention-4 CuTe DSL package extracted from
FlashRT's production GROOT N1.7 Thor runtime at commit
`24df793f4fa2d50780aea03b644208c6e0cb4162`.

The package exposes `flash_attn_func`, `flash_attn_varlen_func`, and the
preallocated `forward_static` CUDA Graph entry. It keeps the vendored
implementation in the private `flashrt_fa4` namespace and vendors the required
quack helper subset, so it cannot replace or mutate upstream `flash_attn`
imports.

See [CARD.md](CARD.md) for usage and [VALIDATION.md](VALIDATION.md) for the
release gate.
