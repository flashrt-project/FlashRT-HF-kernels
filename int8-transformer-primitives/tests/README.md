# Tests

Tests cover:

- static INT8 quantization
- dynamic and static rowwise INT8 quantization
- RMSNorm-to-INT8 producers
- rowwise INT8 linear BF16 output
- SiLU-gated INT8 linear BF16 output

The package tests are self-contained and do not import FlashRT.
