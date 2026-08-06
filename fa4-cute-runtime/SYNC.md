# Source Sync

- Upstream FA4 base: FlashRT commit
  `24df793f4fa2d50780aea03b644208c6e0cb4162`.
- PI0.5 Thor delivery sync: `flashrt-pi05-native-thor-delivery`, including the
  dedicated D256 2CTA implementation and SM110 `seqused_k` support.
- Source subtree:
  `csrc/attention/flash_attn_4_src/flashrt_fa4`
- Scope: forward-only SM100-family CuTe DSL implementation.
- Private import namespace: `flashrt_fa4`.

The synced `interface_fwd_sm100.py` and
`sm100_hd256_2cta_fmha_forward.py` SHA-256 values are respectively
`abfef913d3d53dc738f9e3876366dfe7362ca95e97d69c1c63d1692a33360483` and
`ac0f48bce590409fc2dad360569c68539eb897c9c4af8c3dbcfaf911235b0f77`.
The only package-local source transformation is rewriting absolute
`flashrt_fa4` and `quack` imports beneath the builder-owned
`fa4_cute_runtime` module. Kernel math and launch logic are unchanged.

The bundled `quack` helper subset follows the copy vendored by
`huggingface/kernels-community/flash-attn4` at the time of this package's
initial release. It replaces the external `quack-kernels` runtime dependency;
the FA4 source contents otherwise remain byte-identical to the PI0.5 delivery
snapshot.

When syncing, compare the complete vendor subtree, rerun all Thor shape gates,
and validate both CUTLASS DSL 4.4.x (`sm_110a`) and the builder's 4.5.x
(`sm_101a`) alias before updating this record.
