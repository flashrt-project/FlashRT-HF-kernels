# Validation: flashrt-qkv-cache-rope

Required before publishing this package:

1. Source-extension correctness:

   ```bash
   python flashrt-qkv-cache-rope/tests/test_qkv_cache_rope.py --backend source --mode full
   ```

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
