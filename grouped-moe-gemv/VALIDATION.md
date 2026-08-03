# Validation

Required source gate:

```bash
python grouped-moe-gemv/tests/test_grouped_moe_gemv.py --backend source --mode full
```

Expected source total for version 2: `22/22`.

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

Correctness and low-precision quality are intentionally separate contracts.
The source-BF16 comparison is not used to hide or relabel implementation error.
