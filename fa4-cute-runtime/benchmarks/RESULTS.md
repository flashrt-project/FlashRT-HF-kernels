# Results

Source qualification on NVIDIA Thor (SM110), CUDA 13, CUTLASS DSL 4.4.2:

| Shape | FA4 ms | PyTorch SDPA ms | SDPA / FA4 |
|---|---:|---:|---:|
| S=41, H=32/32, D=48, non-causal | 0.024036 | 0.024666 | 1.026x |
| S=277, H=16/16, D=72, non-causal | 0.116277 | 0.086660 | 0.745x |
| S=1024, H=16/8, D=128, causal | 0.097431 | 0.140887 | 1.446x |

The benchmark reports warmed CUDA-event latency. These rows qualify source
behavior and dispatch boundaries; installed-artifact rows must be rerun after
Hub publication before claiming artifact performance. The package does not
claim universal replacement of SDPA. Consumers should retain profile-based
dispatch for shapes where the existing backend is faster.
