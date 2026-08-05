# Source Sync

- FlashRT commit: `24df793f4fa2d50780aea03b644208c6e0cb4162`
- Source subtree:
  `csrc/attention/flash_attn_4_src/flashrt_fa4`
- Scope: forward-only SM100-family CuTe DSL implementation.
- Private import namespace: `flashrt_fa4`.

The bundled `quack` helper subset follows the copy vendored by
`huggingface/kernels-community/flash-attn4` at the time of this package's
initial release. It replaces the external `quack-kernels` runtime dependency;
kernel math sources remain byte-identical to the FlashRT commit above.

When syncing, compare the complete vendor subtree, rerun all Thor shape gates,
and validate both CUTLASS DSL 4.4.x (`sm_110a`) and the builder's 4.5.x
(`sm_101a`) alias before updating this record.
