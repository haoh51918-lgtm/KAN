# v4 runtime closure iterative-coder log

## Iteration 1 (Phase 1/2: TDD execution-closure seam)

- **Score:** 0.43 (lint=not-run, format=not-run, test=0.68, self=0.65; capped by failing tests)
- **Tests:** expected RED, 15 passed and 7 failed
- **Changes:** added v4 public-seam tests for runtime-pin inclusion/tamper, Python executable hashing, complete distribution identity, RECORD drift, and invalid distribution metadata topology
- **Feedback:** failures isolated every requested missing behavior; the protocol constants were explicitly fixed to v4 in the test fixture so v3 path noise could not mask the RED evidence
- **Next:** implement the smallest fail-closed runtime closure

## Iteration 2 (Phase 1/2: implementation and focused green)

- **Score:** 0.79 (lint=1.0, format=0.0, test=1.0, self=0.73)
- **Tests:** 22 passed
- **Lint:** Ruff check passed; Ruff format check identified only local layout changes
- **Changes:** added base-lock `runtime.files` to the fixed workspace closure, complete installed-distribution RECORD inventory, resolved Python executable SHA256, and three locked environment variables
- **Feedback:** real-environment probing found a stale workspace egg-info without RECORD and found vendored RECORD files inside Ray/Setuptools; no package or project exemption was added
- **Next:** preserve strict missing-RECORD rejection, remove the stale generated metadata outside this scope, and distinguish a distribution's top-level RECORD from vendored metadata

## Iteration 3 (Phase 2/2: real-runtime validation and regression)

- **Score:** 0.95 (lint=1.0, format=1.0, test=1.0, self=0.95)
- **Tests:** 27 focused tests passed; 54 governance tests passed
- **Lint:** Ruff check and Ruff format check passed
- **Changes:** required exactly one top-level `.dist-info/RECORD`, added environment-drift and vendored-RECORD regression tests, and admitted only the two audited predecessor observation states (`none`, `pre_development_admission_count_only`)
- **Real runtime:** `/opt/.venv/bin/python` successfully inventoried 479 unique installed distributions after the stale empty egg-info directory was removed; every inventoried distribution had one valid regular RECORD
- **Next:** done; main agent may create runtime pin files/base lock and generate the implementation lock only after its independent closure audit

## Iteration 4 (Phase 1/3: lock-driven artifact acquisition)

- **Score:** 0.70 (lint=1.00, format=0.00, focused_validation=0.80, self=0.75; capped while the wheelhouse was incomplete).
- **Initial failures:** the first formatter check found local wrapping drift; the first verifier assumed every uv artifact carried a size and raised on Torch; a sequential `pip download` was interrupted after sustained low throughput. The workspace root also proved not to be a Git repository, so no Git-based change inventory was claimed.
- **Refinement:** the downloader now derives one compatible, hash-authorized artifact per normalized lock requirement; rejects unsafe/duplicate names, symlinks, directories, and unexpected files; validates exact size when locked and SHA256 always; publishes with `O_EXCL`; and fsyncs each file plus the destination and parent directories.
- **Focused validation:** 223 unique artifacts selected for 223 requirements, with 217 Linux-applicable distributions, 222 wheels, one sdist, and exact total size 4,884,582,030 bytes. Negative checks proved second-publication, unexpected-directory, and symlink rejection.
- **Next:** exercise production AST/MLP paths and close deterministic launcher/import boundaries.

## Iteration 5 (Phase 2/3: production compatibility and deterministic entrypoints)

- **Score:** 0.93 (lint=1.00, format=1.00, focused_validation=1.00, self=0.90).
- **Compatibility:** both A800 GPUs completed the production E3 profile path for 64 miners at the full 300 updates and six production MLP control pairings at the full 300 updates. All hardened AST values/support masks matched exact production replay. The publication gate compared canonical CPU replay against an independent CPU checkpoint replay at `rtol=0`, `atol=0`, with zero mismatch over 2,874 supported values.
- **Diagnostic:** GPU-training versus CPU-publication predictions differed only at floating roundoff scale (3/2,838 values; maximum absolute difference `1.1102230246251565e-16`) and were identical across the two GPUs; this non-gating diagnostic is explicitly separated from the exact publication gate.
- **Launcher/import closure:** replaced the legacy CLI passthrough with direct `run_s2a_v2_mining` and `run_s2a_v2_development` calls; forced exact project `PYTHONPATH`, bytecode suppression, deterministic Torch/environment state, symlink-safe tracking CWD, and no `MLFLOW_TRACKING_URI`. Qlib's configured experiment-manager URI resolves to the designated `mlruns`, while `C.registered` remains false. Import smoke covers both production pipelines, the decision assembler, E3/MLP/evaluation seams, and the pinned Quanta runner without provider initialization or label access.
- **Next:** prove a network-free fresh rebuild and publish categorized manifests last.

## Iteration 6 (Phase 3/3: offline reconstruction, regression, and manifest seal)

- **Score:** 0.98 (lint=1.00, format=1.00, focused_validation=1.00, full_regression=1.00, self=0.96).
- **Offline reconstruction:** a fresh `/tmp` Python 3.12.3 venv installed all 217 distributions strictly from `wheelhouse/` using `--offline --no-index --no-cache --require-hashes`; `uv pip check` passed all 217. The rebuilt runtime asserted Torch 2.9.1+cu129, CUDA 12.9, cuDNN 9.10.2.21, and two visible GPUs.
- **Regression:** after final live-runtime lock hardening, the main-agent-owned exact-v4 regression passed 368 tests with 933 warnings in 129.95 seconds. Ruff checks passed for `src tests` and runtime tools; the seven active Python files passed format check. The 47-file historical project format drift remained outside this surgical gate.
- **Manifest seal:** acquisition-only staging/fallback copies and tool bytecode caches were removed. Exclusive-create categorized JSON/TSV manifests bind 20 runtime files (5 frozen roots, 5 closure tools, 10 evidence files), all 223 wheelhouse artifacts, and all 217 installed `.dist-info/RECORD` files. Independent post-generation hash/size walks passed.
- **Boundary:** no scientific source, test, configuration, preregistration, governance decision, real labels, Qlib provider initialization, mining output, or development result was changed or opened by runtime closure work.
- **Next:** parent may bind the final launcher/helper identities in the v4 base and implementation locks.
