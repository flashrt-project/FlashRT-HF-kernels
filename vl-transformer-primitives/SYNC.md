# Source Sync

Source kernels are adapted from FlashRT:

- `official/FlashRT/csrc/kernels/qwen3_qkv_post_proc.cu`
- `official/FlashRT/csrc/kernels/qwen3_qkv_post_proc.cuh`
- `official/FlashRT/csrc/kernels/norm.cu` vision-token pooling logic

The Hub-facing wrapper exposes tensor APIs and shape checks suitable for
`kernels` / `kernel-builder`; it does not expose FlashRT serving-internal
pointer APIs.
