# Validation

Release gates are run on an SM120a RTX 5090 against FP32-softmax PyTorch SDPA
using the same centered Q/K/V tensors. The test reports cosine similarity for
every row and requires at least 0.97. This is a speed-first FP4 tier; it is not
held to the SageAttention2 accuracy contract.

The full matrix covers D64 and D128, both `per_block_mean` modes, video lengths
6144 and 24576, and audio length 2688. Every case also checks caller-owned
pointer stability and bitwise CUDA Graph replay.

```bash
python sageattention3-blackwell/tests/test_sageattention3_blackwell.py \
  --backend source --mode full
python sageattention3-blackwell/tests/test_sageattention3_blackwell.py \
  --backend installed --artifact <build/variant> --mode full
```

Performance must be reported against PyTorch SDPA and SageAttention2 on the
same device and tensor contract. Packaged-artifact numbers replace source-gate
numbers only after cold-load validation from Kernel Hub.
