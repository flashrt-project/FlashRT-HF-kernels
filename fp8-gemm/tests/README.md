# Tests

```bash
python fp8-gemm/tests/test_fp8_gemm.py --backend source --mode full
```

SM110 full mode adds PI0.5, GROOT, Cosmos Edge, and LingBot VLA projection
shapes plus forced Sq/T1/Wide correctness rows. Use `--backend installed` with
the exact artifact directory for the release gate.
