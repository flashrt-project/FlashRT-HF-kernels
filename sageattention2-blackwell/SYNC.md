# Source synchronization

The SageAttention2 CUDA implementation is synchronized against FlashRT PR
[#172](https://github.com/flashrt-project/FlashRT/pull/172), source commit
`f57459d1746128e81e5fd9e8d8173034f67a255c`.

The package preserves the native quantization expressions and attention core.
The FP8-V convenience path also preserves the native two-stage coalesced
transpose/pad/permutation followed by vectorized per-channel FP8 quantization.
Permitted package-only differences are:

- Tensor/custom-op bindings and caller-owned workspace APIs;
- capability metadata, tests, examples, and benchmark harnesses;
- the partial final Q-tile bounds fix for `seqlen_q % 128 in [1, 96]`.

Do not replace the CUDA core with a separate reimplementation. Future source
updates must first be diffed against the pinned FlashRT implementation, then
rerun partial-tile, CUDA Graph, and native-core parity gates.
