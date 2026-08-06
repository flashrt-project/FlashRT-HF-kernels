# Validation

Required source gate:

```bash
python grouped-moe-gemv/tests/test_grouped_moe_gemv.py --backend source --mode full
```

Expected source totals for version 2 are hardware-specific: `26/26` on
SM120/SM121 and `13/13` on SM110, where the SM120-only W4A4 cases are replaced
by explicit rejection and native W4A16 graph/routing gates.

The full gate covers:

- version 1 W4A16 API regression;
- W4A4 `K%16` contract, including non-MMA `K=80`;
- decode/verify gate-up layouts (`M=1/7, top_k=8`);
- decode/verify down layouts (`M=8/56, top_k=1`);
- exact target projections `(N,K)=(1024,2048)` and `(2048,512)`;
- bitwise grouped-vs-per-route native parity;
- max/p99/mean/cosine against the dequantized NVFP4 mathematical reference;
- separately reported NVFP4-vs-source-BF16 quality;
- allocation-free CUDA Graph replay, bitwise replay stability, and device
  `expert_idx` mutation between replays.

Validated source results:

| Device | Backend | Result | Notes |
| --- | --- | ---: | --- |
| RTX 5090 (SM120) | source | 26/26 | W4A16 and W4A4, target MoE shapes, graph replay |
| Jetson AGX Thor (SM110) | source | 13/13 | native edge W4A16, `kUnroll=2`, graph replay, W4A4 fail-fast |

The SM110 backend uses the same FlashRT native CUDA implementation as PR #169.
Release artifacts must additionally pass installed-artifact parity against
this source build; testing only Python symbol presence is not sufficient.

Correctness and low-precision quality are intentionally separate contracts.
The source-BF16 comparison is not used to hide or relabel implementation error.
