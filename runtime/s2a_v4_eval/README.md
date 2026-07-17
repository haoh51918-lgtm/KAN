# S2a v4 evaluation runtime

This directory is the isolated, offline-rebuildable S2a v4 evaluation
runtime. It does not reuse or concatenate another environment's
`site-packages`.

The declared scientific root versions and the `pytest`/`ruff` development
tools are fixed in `pyproject.toml`. `uv.lock` is the resolver lock,
`requirements.lock` is the installable hash-locked export, and `wheelhouse/`
contains the corresponding local artifacts.

The closure contains 223 hash-verified artifacts (222 wheels and one sdist)
for 217 Linux-applicable distributions, totalling 4,884,582,030 bytes. The
six additional artifacts preserve platform-marker completeness. Categorized
machine and human manifests are in `manifests/`.

Run all commands below from the repository root. Verify the authoritative
wheelhouse before use:

```bash
runtime/s2a_v4_eval/.venv/bin/python \
  runtime/s2a_v4_eval/tools/fetch_wheelhouse.py \
  runtime/s2a_v4_eval/wheelhouse --verify-only
```

Rebuild without an index or network access:

```bash
uv venv --python runtime/s2a_v4_eval/.venv/bin/python /tmp/s2a_v4_offline
uv pip install --offline --no-cache --no-index \
  --find-links runtime/s2a_v4_eval/wheelhouse \
  --require-hashes -r runtime/s2a_v4_eval/requirements.lock \
  --python /tmp/s2a_v4_offline/bin/python
uv pip check --python /tmp/s2a_v4_offline/bin/python
```

The deterministic launcher re-executes itself with the frozen environment,
an exact project-only `PYTHONPATH`, and deterministic Torch settings before
importing mining or evaluation code. Its mining and development commands call
the v4 production pipeline directly:

```bash
runtime/s2a_v4_eval/.venv/bin/python \
  runtime/s2a_v4_eval/tools/runtime_launcher.py --workspace . lock-verify
runtime/s2a_v4_eval/.venv/bin/python \
  runtime/s2a_v4_eval/tools/runtime_launcher.py --workspace . mining
runtime/s2a_v4_eval/.venv/bin/python \
  runtime/s2a_v4_eval/tools/runtime_launcher.py --workspace . development
```

Development establishes
`evaluations/runtime/s2a_v4_tracking` as the tracking working directory
before the first Qlib import. Inspect the non-initializing receipt and import
closure with:

```bash
runtime/s2a_v4_eval/.venv/bin/python \
  runtime/s2a_v4_eval/tools/runtime_launcher.py --workspace . tracking-receipt
PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/zju_0012/htq/aaai26_alpha/aaai27_evosci/src \
  runtime/s2a_v4_eval/.venv/bin/python \
  runtime/s2a_v4_eval/tools/import_audit.py
```
