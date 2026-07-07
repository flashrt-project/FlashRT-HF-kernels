# Source Sync

- Upstream: `official/FlashRT`
- Source commit at initial sync: see repository parent checkout at packaging time.
- Upstream source:
  - `csrc/kernels/qwen36_misc.cu`
  - `csrc/kernels/qwen36_misc.cuh`
- Package-local files:
  - `csrc/speculative_draft_primitives.cu`
  - `csrc/speculative_draft_primitives.cuh`

Local edits:

- Extracted only BF16 logits argmax and speculative accepted-prefix kernels.
- Renamed namespace and public APIs to model-neutral names.
- Removed unrelated embedding, RoPE, TurboQuant, and transformer helper code.
- Added Tensor validation and Torch custom-op registration in `torch-ext/`.

Runtime constraints:

- CUDA contiguous tensors only.
- BF16 logits, int64 draft/argmax tensors, int32 accept counter.
- `parts <= 128` for partitioned reduction.
