# Results

Source qualification on NVIDIA Thor (SM110), CUDA 13, CUTLASS DSL 4.4.2:

| Shape | FA4 ms | PyTorch SDPA ms | SDPA / FA4 |
|---|---:|---:|---:|
| S=41, H=32/32, D=48, non-causal | 0.024036 | 0.024666 | 1.026x |
| S=277, H=16/16, D=72, non-causal | 0.116277 | 0.086660 | 0.745x |
| S=1024, H=16/8, D=128, causal | 0.097431 | 0.140887 | 1.446x |

PI0.5 D256 qualification is intentionally pending the target-host rerun:

| Shape | FA4 ms | PyTorch SDPA ms | SDPA / FA4 |
|---|---:|---:|---:|
| Sq/Sk=320, H=8/1, D=256 | pending | pending | pending |
| Sq=456, Sk=968, valid_k=456, H=8/1, D=256 | pending | pending | pending |
| Sq=712, Sk=968, valid_k=712, H=8/1, D=256 | pending | pending | pending |
| Sq/Sk=968, H=8/1, D=256 | pending | pending | pending |

The benchmark reports warmed CUDA-event latency. These rows qualify source
behavior and dispatch boundaries; installed-artifact rows must be rerun after
Hub publication before claiming artifact performance. PI0.5 rows must also be
compared in the same process with the delivery-native FA4 path. The package does not
claim universal replacement of SDPA. Consumers should retain profile-based
dispatch for shapes where the existing backend is faster.
