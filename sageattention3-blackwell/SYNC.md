# Source synchronization

The CUDA/CuTe implementation is packaged from
[`thu-ml/SageAttention`](https://github.com/thu-ml/SageAttention), commit
`d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`, directory
`sageattention3_blackwell`, under Apache-2.0.

Package adaptations are limited to:

- removal of upstream Python-extension module registration in favor of HF
  kernel-builder custom ops;
- NHD input adapters that write the native packed HND/transpose layouts;
- caller-owned output, LSE, semaphore, and quantization workspaces;
- explicit device, shape, layout, dtype, and unsupported-GQA errors;
- tests, capability metadata, examples, and benchmark harnesses.

The blockscaled attention templates, tile schedule, and FP4 conversion logic
remain the upstream implementation. Any upstream refresh must rerun D64/D128,
both block-mean modes, 6K/24K video, 2.7K audio, graph replay, and SDPA cosine
gates before release.
