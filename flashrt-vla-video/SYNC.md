# Source Sync

Upstream source: `../official/FlashRT`

This package tracks the original v1 VLA/video QKV postprocess slice:

- decode Q RMSNorm + rotate-half RoPE staging
- decode K RMSNorm + rotate-half RoPE + K/V cache slot write
- packed QKV split + Q/K RMSNorm + interleaved RoPE

Long-term maintenance note:

- Keep this package scoped to reusable VLA/video attention postprocess kernels.
- Put highly specific QKV/cache runtime APIs in `flashrt-qkv-cache-rope` when a
  narrower package name makes usage clearer.
- Put spatiotemporal/world-model layout helpers in
  `flashrt-spatiotemporal-layout`.
- If the namespace is reorganized later, do it as an explicit migration plan
  rather than an accidental in-place rename.
