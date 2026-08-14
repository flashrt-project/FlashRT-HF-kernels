# Validation

Release gates are run on an SM120a RTX 5090 against FP32-softmax PyTorch SDPA
using the same centered Q/K/V tensors. The test reports cosine similarity for
every row and requires at least 0.97. This is a speed-first FP4 tier; it is not
held to the SageAttention2 accuracy contract.

The CUDA 13 full matrix covers D64 and D128, both `per_block_mean` modes, video
lengths 6144 and 24576, and audio length 2688. Every case also checks
caller-owned pointer stability and bitwise CUDA Graph replay. The CUDA 12.8
matrix covers D64 and explicitly verifies that D128 is rejected at both the
Python capability boundary and the compiled operator boundary. CUDA 12.8 D128
is excluded because compiler-generated local-memory spills make it slower than
SDPA, not because correctness tolerance was relaxed.

The fused raw-BF16 API is compared directly with the former two-stage contract.
Acceptance requires cosine >= 0.99999, with aligned cases expected to be
bitwise equal. The matrix includes an unaligned `S=5070` case to validate
internal zero padding and output cropping. The RTX 5090 performance gate is
all-in latency <= 13.0 ms for `B=1,H=32,S=24576,D=128`; allocation is excluded,
but centering, quantization, correction GEMM and attention are all included.

```bash
python sageattention3-blackwell/tests/test_sageattention3_blackwell.py \
  --backend source --mode full
python sageattention3-blackwell/tests/test_sageattention3_blackwell.py \
  --backend installed --artifact <build/variant> --mode full
```

Performance must be reported against PyTorch SDPA and SageAttention2 on the
same device and tensor contract. Packaged-artifact numbers replace source-gate
numbers only after cold-load validation from Kernel Hub.

The current cold-loaded CUDA 13 artifact measures 13.111 ms at the long-video
gate, so the strict 13.0 ms target remains open by 0.111 ms. It is still
reported as a miss; no tolerance or timing scope is changed to turn it into a
pass.
