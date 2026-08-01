# Source synchronization

The implementation is synchronized from:

- FlashRT revision `8fdd8326edf44fc3e7ad99786ea80c070fd984d5`

- `official/FlashRT/flash_wm/csrc/cutlass_fp8_fmha.cu`
- CUTLASS example 77 `device/`, `kernel/`, and `collective/` headers
- FlashRT's modified FP8-rescale mainloop collective

Hub adaptation removes process-global CUDA allocations and accepts caller-owned
LSE/workspace tensors. The public API is model-neutral. Vendored CUTLASS
headers received whitespace-only normalization, and the non-causal mainloop
uses CUTLASS `ResidualMask` so non-128 KV tails are processed correctly.
