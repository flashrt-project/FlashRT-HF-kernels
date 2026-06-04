# Scripts

Repository-level helper scripts for the v1 batch.

Current scripts:

- `correctness_audit.py`: blocks release builds while known accuracy gaps are
  still documented.
- `accuracy_sweep.py`: runs source or installed-package accuracy sweeps over
  the v1 shape grids. It records max error, mean error, p99 error, relative
  error, cosine similarity, dtype, and tolerance per check. Use
  `--backend source --mode full` before the release build window and
  `--backend installed --mode full` after artifacts are installed or copied.
- `prebuild_check.py`: checks v1 package structure, `build.toml` source lists,
  tracked internal directories, stale build artifacts, and optionally
  `kernel-builder-docker check-config`.
- `release_build_plan.py`: prints the release-candidate build-window command
  sequence; it only executes when called with `--execute`.
- `run_built_artifact_benchmarks.py`: local release-candidate runner for the
  public `kernels.benchmark.Benchmark` scripts against either local source
  extensions or copied built artifacts. Add `--compile-baseline` to time
  `torch.compile` versions of benchmark reference functions when they are
  available and verified equivalent to the eager reference. The runner fails
  fast on verification or execution failures by default. Only use
  `--allow-diagnostic-failures` for internal triage runs that intentionally
  record failed rows as `nan`.

`torch.compile` baselines are not automatically fair just because compilation
succeeds. For quantized references, especially FP8/FP4 fake-quant chains, the
compiled reference must first match the eager reference. If Inductor changes
rounding behavior at a quantization boundary, mark that compiled baseline as
unsupported and report eager correctness plus other valid baselines instead.

Example:

```bash
python scripts/prebuild_check.py --check-config
python scripts/accuracy_sweep.py \
  --backend source \
  --mode full \
  --package all \
  --output-json internal-tests/correctness/source-full.json \
  --output-md internal-tests/correctness/source-full.md
python scripts/correctness_audit.py
python scripts/release_build_plan.py
python scripts/run_built_artifact_benchmarks.py --backend source --package all --compile-baseline
python scripts/run_built_artifact_benchmarks.py --backend artifact --package all --compile-baseline
```

See `docs/release-runbook.md` for the full build-window procedure.
