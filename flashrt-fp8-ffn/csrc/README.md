# csrc

CUDA source for FlashRT FP8 FFN Tensor API wrappers.

Dimensions divisible by four use vectorized BF16 loads and packed FP8 stores
for input quantization and the GELU/quantization producer. The final BF16 bias
epilogue is vectorized under the same alignment contract. Other dimensions use
the scalar fallback. FP8 GEMMs retain the cached cuBLASLt descriptor path after
package-local tile candidates were benchmarked and rejected.
