# Validation

Release requires FP16 and BF16 parity against PyTorch SDPA at key lengths
41, 277, 1024, 1025, and 2048; poisoned padded scratch; CUDA Graph replay;
source and installed-artifact runs on SM110; and native-vs-package timing.
