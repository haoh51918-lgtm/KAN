# V4 lineage-size migration — More Effort iteration log

## Scope

- Authority: frozen v4 protocol configuration only.
- Migrated boundary: `admission.minimum_library_size` through `admission.library_cap`.
- Preserved unchanged: IC, coverage, profile quota, diversity, KAN/GP/MLP budgets, seeds, training steps, development dates, Quanta settings, and promotion criteria.
- Active code contains no new literal v4 floor; the floor is read from the frozen config and passed explicitly across module boundaries.

## Iteration 1 — Runner regression (RED)

- **Score**: 0.36 (lint=not run, format=not run, test=0.0, self=0.6)
- **Tests**: failed as expected (0/5); every arm stopped at the stale mining-top `[8, 16]` check.
- **Changes**: parameterized the runner fixture and added a six-factor, five-arm cross-link regression.
- **Feedback**: the stale lower bound existed in mining lineage, factor-library replay, and matched-blackbox replay.
- **Next**: replace all three checks with the same frozen-admission bounds.

## Iteration 2 — Runner implementation (GREEN)

- **Score**: 0.86 (lint=1.0, format=0.0, test=1.0, self=0.9)
- **Lint**: passed.
- **Format**: one newly added line required formatting.
- **Tests**: passed (34/34 runner tests after adding malformed-bound cases).
- **Changes**: added strict `_library_size_bounds(config)` and reused it in all three runner cross-links.
- **Feedback**: evaluation-layer migration was correct, but upstream mining and publication paths still carried stale lower bounds.
- **Next**: extend the same config-derived boundary through the active v4 pipeline.

## Iteration 3 — Mining and MLP contracts (RED → GREEN)

- **Score**: 0.96 (lint=1.0, format=1.0, test=1.0, self=0.9)
- **Tests**: initial failures reproduced stale selection and MLP pairing contracts; final 20/20 passed.
- **Changes**:
  - production selection now relies on its explicit frozen min/cap inputs;
  - size-matched GP and permutation APIs receive explicit frozen min/cap;
  - MLP pairing validation and execution receive explicit frozen min/cap;
  - `v2_pipeline` passes both values from `inputs.protocol.admission`.
- **Feedback**: six is accepted, five is incomplete/rejected, and a seventeenth candidate is capped or rejected according to the protocol boundary.
- **Next**: migrate publication builders and top-level cross-links.

## Iteration 4 — Artifact builders (RED → GREEN)

- **Score**: 0.96 (lint=1.0, format=1.0, test=1.0, self=0.9)
- **Tests**: initial public-builder call failed on the missing boundary interface; final 8/8 passed.
- **Changes**:
  - factor library, MLP control, and mining top builders receive explicit min/cap;
  - one shared fail-closed helper validates library, selection, budget, and all six child counts;
  - `_publish_mining` reads and validates frozen admission once, then forwards it to every builder.
- **Feedback**: the complete six-factor top bundle succeeds; five and seventeen fail before staging publication.
- **Next**: migrate decision assembly and verify duplicate policy fields cannot drift.

## Iteration 5 — Decision assembler (RED → GREEN)

- **Score**: 0.96 (lint=1.0, format=1.0, test=1.0, self=0.9)
- **Tests**: initial 0/4 reproduced stale inventory checks; final 4/4 new cases and 43/43 runner/assembler regressions passed.
- **Changes**:
  - assembler reads min/cap from frozen admission;
  - scoring ledgers and published inventories use the same bounds;
  - `s2a_decision.integrity.production_library_size_minimum` must equal the admission minimum.
- **Feedback**: complete six-factor evidence succeeds; counts five and seventeen and a divergent integrity floor fail closed.
- **Next**: integrated focused verification.

## Iteration 6 — Integrated verification

- **Score**: 0.96 (lint=1.0, format=1.0, test=1.0, self=0.9)
- **Lint**: passed on all 11 changed source/test files.
- **Format**: passed on all 11 changed source/test files.
- **Tests**: 77 focused and related tests passed:
  - mining selection and MLP: 20;
  - mining pipeline: 6;
  - runner and decision assembler: 43;
  - artifact bundle: 8.
- **Warnings**: only the existing pandas `FutureWarning` from DSL lagged-support evaluation.
- **Result**: ready for full-suite verification before the v4 implementation lock.

## Files

- `src/mirage_kan/mining/v2_scoring.py`
- `src/mirage_kan/mining/mlp_control.py`
- `src/mirage_kan/mining/v2_pipeline.py`
- `src/mirage_kan/artifacts/v2_bundle.py`
- `src/mirage_kan/evaluation/v2_runner.py`
- `src/mirage_kan/evaluation/v2_decision_assembler.py`
- `tests/mining/test_v2_scoring.py`
- `tests/mining/test_mlp_control.py`
- `tests/artifacts/test_v2_bundle.py`
- `tests/evaluation/test_v2_runner.py`
- `tests/evaluation/test_v2_decision_assembler.py`

## Quanta preflight follow-up — deterministic arm execution

### Iteration 7 — Concurrency regression (RED)

- **Score**: 0.36 (lint=not run, format=not run, test=0.0, self=0.6)
- **Tests**: failed as expected (0/2).
- **Observed risk**: all five arms entered staging before the first completed, and an arm-three failure still allowed arms four and five to start.
- **Root cause**: a five-worker thread pool ran adapter import and global `qlib.init` concurrently even though exact backtest capture serialized the scientific invocation itself.
- **Next**: execute the frozen `ARMS` order directly and preserve cleanup of completed or invalid staging bundles.

### Iteration 8 — Sequential execution (GREEN)

- **Score**: 0.96 (lint=1.0, format=1.0, test=1.0, self=0.9)
- **Changes**:
  - removed `Future`, `ThreadPoolExecutor`, and `as_completed`;
  - staged arms in the frozen tuple order;
  - retained per-arm validation, cleanup of the current invalid staging result, cleanup of all previously completed staging results, and propagation of the original exception.
- **Focused tests**: 22/22 pipeline and Quanta-adapter tests passed.
- **Related regression**: 90/90 evaluation tests passed.
- **Lint/format**: both changed files passed Ruff check and format verification.
- **Result**: adapter import, Qlib global initialization, MLflow recording, exact report capture, and Quanta evaluation can no longer overlap across arms in one development process.

### Additional files

- `src/mirage_kan/evaluation/v2_pipeline.py`
- `tests/evaluation/test_v2_pipeline.py`

### Iteration 9 — Authority postfix and cleanup regressions (RED)

- **Score**: 0.36 (lint=not run, format=not run, test=0.0, self=0.6)
- **Tests**: the two authority-order tests failed as expected because all five arm capabilities were issued before the first arm started.
- **Additional reproduced failure**: an exception from `_remove_staging` replaced the original arm failure and stopped cleanup of later completed staging directories.
- **Boundary proof added**: runner workspaces with five or seventeen outputs must fail against frozen admission bounds.
- **Next**: issue one arm capability immediately before its stage call and make cleanup best-effort without replacing the primary exception.

### Iteration 10 — Authority postfix and best-effort cleanup (GREEN)

- **Score**: 0.96 (lint=1.0, format=1.0, test=1.0, self=0.9)
- **Changes**:
  - removed the pre-issued capability mapping;
  - moved `guard.revalidate(...)` inside the frozen arm loop and immediately before `stage_v2_arm`;
  - on arm failure, attempts cleanup for the current returned staging bundle and every previously completed bundle;
  - cleanup failures are attached to the original error with `add_note`, while the original arm or authority exception remains primary.
- **Focused tests**: 59/59 pipeline, runner, and Quanta-adapter tests passed.
- **Related regression**: 93/93 evaluation tests passed.
- **Lint/format**: all three changed files passed Ruff check and format verification.
- **Result**: authority receipts are issued only for arms that actually start; failure cannot leave unused later-arm receipts, and cleanup errors cannot hide the scientific/control-arm failure.

### Postfix test file

- `tests/evaluation/test_v2_runner.py`
