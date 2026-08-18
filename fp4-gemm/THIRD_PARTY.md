# Third-Party Sources

The small-M W4A16 implementation under `csrc/marlin/` is adapted from the
Apache-2.0 Marlin backend in vLLM commit
`41f179b57aa8ab6f634f508128ce1f1efadd0eb1`:

- https://github.com/vllm-project/vllm
- https://github.com/IST-DASLab/marlin

Vendored source files retain their upstream copyright and Apache-2.0 license
headers. FlashRT adds a restricted BF16 x NVFP4 configuration selector,
bind-time ModelOpt weight adoption, a caller-owned workspace/output Tensor API,
and Kernel Hub build/test integration. The resulting package has no runtime
dependency on vLLM.
