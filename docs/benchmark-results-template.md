# Benchmark Results Template

Use this template when refreshing package-level `benchmarks/RESULTS.md` files
or hardware-specific reports under `internal-tests/`.

## Environment

| Field | Value |
| --- | --- |
| Package | `<flashrt-package>` |
| Kernel artifact | `source-extension`, `built-artifact:<variant>`, or `hub:<repo-id>@<version>` |
| GPU | `<name>` |
| Compute capability | `<sm>` |
| Driver | `<driver>` |
| CUDA runtime | `<cuda>` |
| PyTorch | `<version>` |
| Kernels package | `<version or git sha>` |
| FlashRT HF kernels commit | `<git sha>` |
| Warmup | `<count>` |
| Iterations | `<count>` |

## Summary

| Kernel | Shape families | Best public label | Main baseline | Speedup range | Notes |
| --- | --- | --- | --- | ---: | --- |
| `<kernel>` | `<families>` | `headline/support/compatibility` | `<baseline>` | `<x-y>x` | `<notes>` |

## Detailed Results

| Kernel | Shape family | Shape | Tile/dispatch | FlashRT ms | Eager ms | vs eager | Compile ms | vs compile | Strong baseline ms | vs strong | Correctness | Label | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `<kernel>` | `<family>` | `<shape>` | `<tile>` | `<ms>` | `<ms>` | `<x>` | `<ms or n/a>` | `<x or n/a>` | `<ms or n/a>` | `<x or n/a>` | `<max error / byte parity>` | `<label>` | `<notes>` |

Only fill `Compile ms` and `vs compile` after the compiled reference output has
been verified against the eager reference. If the compiled reference is not
numerically equivalent, use `n/a` and record the unsupported reason in `Notes`.

## Missing Baselines

| Kernel | Missing baseline | Reason | Next action |
| --- | --- | --- | --- |
| `<kernel>` | `<torch.compile / CUTLASS / cuBLASLt / unfused chain>` | `<reason>` | `<action>` |

## Reproduction

```bash
# Correctness
python scripts/accuracy_sweep.py --backend <source|installed|hub> --mode full --package <package>

# Built artifact benchmark with compiled PyTorch reference where possible
python scripts/run_built_artifact_benchmarks.py \
  --package <package> \
  --variant <variant> \
  --compile-baseline \
  --warmup 10 \
  --iterations 50 \
  --output internal-tests/<package>/built-artifact-results.json
```

## Promotion Decision

| Decision | Meaning |
| --- | --- |
| Promote | Rows can be used in root README or package card |
| Hold | Correct but missing required baseline |
| Retune | Correct but too slow for target shape |
| Drop | Not useful for v1 public package |

Decision: `<Promote/Hold/Retune/Drop>`

Rationale: `<short rationale>`
