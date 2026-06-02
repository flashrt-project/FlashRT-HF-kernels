# Scripts

Repository-level helper scripts for the v1 batch.

Current scripts:

- `prebuild_check.py`: checks v1 package structure, `build.toml` source lists,
  tracked internal directories, stale build artifacts, and optionally
  `kernel-builder-docker check-config`.
- `release_build_plan.py`: prints the full v1 build-window command sequence;
  it only executes when called with `--execute`.

Example:

```bash
python scripts/prebuild_check.py --check-config
python scripts/release_build_plan.py
```

See `docs/release-runbook.md` for the full build-window procedure.
