# Scripts

Repository-level helper scripts for the v1 batch.

Current scripts:

- `prebuild_check.py`: checks v1 package structure, `build.toml` source lists,
  tracked internal directories, stale build artifacts, and optionally
  `kernel-builder-docker check-config`.

Example:

```bash
python scripts/prebuild_check.py --check-config
```
