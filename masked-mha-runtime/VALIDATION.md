# Validation

Release requires FP16 and BF16 parity against PyTorch SDPA at key lengths
41, 277, 1024, 1025, and 2048; poisoned padded scratch; CUDA Graph replay;
source and installed-artifact runs on SM110; and native-vs-package timing.

On 2026-08-06, source full passed `10/10` on NVIDIA Thor (SM110), PyTorch
`2.13.0+cu130`, and CUDA 13.0. The additive `forward_seqused_static` rows cover
PI0.5 `(Sq,H,D)=(10,8,256)`, `Sk_max=456/968`, and device-resident
`valid_k=456/712/968`. Worst p99 absolute error was `0.000122`, worst cosine
was `0.99999982`, and graph replay was bitwise deterministic.
