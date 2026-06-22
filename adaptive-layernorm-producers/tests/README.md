# Tests

Run correctness tests from the repository root:

```bash
python adaptive-layernorm-producers/tests/test_adaptive_layernorm_producers.py --backend source --mode full
```

After Kernel Hub build/publish, run installed-artifact validation:

```bash
python adaptive-layernorm-producers/tests/test_adaptive_layernorm_producers.py --backend installed --mode full
```

The source mode compiles the package locally with `torch.utils.cpp_extension`.
The installed mode imports the package exactly as exposed by Kernel Hub.
