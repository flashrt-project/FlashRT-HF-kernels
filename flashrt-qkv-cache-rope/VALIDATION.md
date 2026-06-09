# Validation: flashrt-qkv-cache-rope

Required before publishing this package:

1. Source-extension correctness:

   ```bash
   python flashrt-qkv-cache-rope/tests/test_qkv_cache_rope.py --backend source --mode full
   ```

   The full mode includes the GQA sequence cache-write API
   `qkv_split_rope_kvcache_bf16` with a PI0.5 decoder-shaped row
   `(B=1, S=10, q_heads=8, kv_heads=1, head_dim=256)`, a batched GQA row,
   cache prefix/suffix preservation checks, and invalid shape/bounds rejection.

2. Source-extension benchmark:

   ```bash
   python flashrt-qkv-cache-rope/benchmarks/benchmark.py \
     --backend source \
     --shapes all \
     --warmup 3 \
     --iters 10
   ```

3. Kernel-builder artifact build:

   ```bash
   kernel-builder build-and-copy flashrt-qkv-cache-rope
   ```

4. Built-artifact correctness:

   ```bash
   PYTHONPATH=<artifact-path> \
   python flashrt-qkv-cache-rope/tests/test_qkv_cache_rope.py \
     --backend installed \
     --mode full
   ```

5. Built-artifact benchmark:

   ```bash
   python flashrt-qkv-cache-rope/benchmarks/benchmark.py \
     --backend installed \
     --artifact <artifact-path> \
     --shapes all
   ```

6. Multi-hardware matrix:

   Add hardware claims only after the same correctness and benchmark commands
   pass on that machine.

## Local validation log

2026-06-09:

- Source correctness passed:

  ```bash
  python flashrt-qkv-cache-rope/tests/test_qkv_cache_rope.py \
    --backend source \
    --mode full
  ```

- Source smoke benchmark passed in `pi0-stablehlo-test` with the isolated
  `internal-tests/envs/openpi-baseline` venv. The PI0.5 GQA cache row reported
  exact Q/K/V agreement and `4.998 us` FlashRT time.

- Kernel-builder config, build, build check, and ABI check passed:

  ```bash
  kernel-builder-docker check-config .
  kernel-builder-docker build --variant torch212-cxx11-cu132-x86_64-linux .
  kernel-builder-docker build --variant torch211-cxx11-cu128-x86_64-linux .
  kernel-builder-docker check-builds .
  kernel-builder-docker check-abi .
  ```

- Built-artifact correctness passed with
  `build/torch211-cxx11-cu128-x86_64-linux` in the matching
  `.flashrt-hub-smoke-torch211` venv:

  ```bash
  PYTHONPATH=flashrt-qkv-cache-rope/build/torch211-cxx11-cu128-x86_64-linux \
  python flashrt-qkv-cache-rope/tests/test_qkv_cache_rope.py \
    --backend installed \
    --mode full
  ```

- Built-artifact benchmark passed with
  `build/torch211-cxx11-cu128-x86_64-linux`; results are recorded in
  `benchmarks/RESULTS.md`. The PI0.5 GQA cache row reported exact Q/K
  agreement for `(B=1, S=10, q_heads=8, kv_heads=1, head_dim=256)` and
  `3.802 us` FlashRT time.

- `build/torch212-cxx11-cu132-x86_64-linux` imports fail in the
  `pi0-stablehlo-test` Torch 2.9 nightly venv with a Torch C++ ABI undefined
  symbol. This is an environment/variant mismatch; `check-abi` passes for the
  artifact, and the torch211/cu128 artifact passes in its matching venv.
