# S0 Iterative Implementation Log

> Ruff is not installed in the project environment; lint and format checks are skipped rather than installing an unrequested dependency. `/usr/bin/python -m pytest` remains mandatory.

## Iteration 1 (Phase 1/4: PIT and typed DSL contract)

- **Score**: 0.76 (lint=skipped format=skipped test=0.86 self=0.70)
- **Lint**: skipped (`ruff` not installed)
- **Format**: skipped (`ruff` not installed)
- **Tests**: failed at collection (6 behavior tests passed; publication module not implemented yet)
- **Changes**: `pyproject.toml`, `src/mirage_kan/data/pit.py`, `src/mirage_kan/dsl/core.py`, data/DSL tests
- **Feedback**: PIT and DSL phase tests pass. Full suite correctly exposes the next missing dependency: immutable library publication.
- **Next**: next_phase

## Iteration 2 (Phase 2/4: immutable factor-library publication)

- **Score**: 0.80 (lint=skipped format=skipped test=0.88 self=0.75)
- **Lint**: skipped (`ruff` not installed)
- **Format**: skipped (`ruff` not installed)
- **Tests**: failed at collection (7 behavior tests passed; Quanta adapter not implemented yet)
- **Changes**: `src/mirage_kan/artifacts/library.py`, publication behavior test promoted to passing
- **Feedback**: Atomic Linux no-replace publication, file hashes, AST records, support hashes, and exact raw-data recomputation pass. Full suite now exposes only the planned adapter dependency.
- **Next**: next_phase

## Iteration 3 (Phase 3/4: pinned Quanta adapter)

- **Score**: 0.92 (lint=skipped format=skipped test=1.00 self=0.86)
- **Lint**: skipped (`ruff` not installed)
- **Format**: skipped (`ruff` not installed)
- **Tests**: passed (10/10)
- **Changes**: `src/mirage_kan/evaluation/quanta.py`, evaluator identity/wiring tests promoted to passing
- **Feedback**: Adapter rejects source drift before import and invokes the real computed-factor dataset plus real training/backtest methods with no proxy evaluator. Baseline link preserves early-stopping semantics.
- **Next**: next_phase

## Iteration 4 (Phase 4/4: real cache publication and Quanta evaluation)

- **Score**: 0.96 (lint=skipped format=skipped test=1.00 self=0.93)
- **Lint**: skipped (`ruff` not installed)
- **Format**: skipped (`ruff` not installed)
- **Tests**: passed (11/11)
- **Changes**: frozen data/evaluator configs, CLI and seed library, GPFS manifest-last publication, real audit/library/evaluation artifacts, attempt ledger
- **Feedback**: Exact label values and the membership/observation split passed on the real cache. Linux `renameat2(RENAME_NOREPLACE)` was unsupported by GPFS and was replaced with exclusive mkdir/O_EXCL/manifest-last/fsync publication. The first Quanta attempt stopped before training because current MLflow rejects filesystem tracking by default; retrying with its documented `MLFLOW_ALLOW_FILE_STORE=true` compatibility switch completed real 500-round LightGBM and 966-step cost-aware TopkDropout backtest. Published code identity still matches the current source tree.
- **Next**: done

# S1 Torch Executor Iteration Log

> The immutable `factor_libraries/seed_ast_v1/manifest.json` binds the published S0
> source tree only. S1 changes are a new working-tree state and do not republish or
> extend that identity. Ruff is not installed; checks remain unavailable rather than
> installing packages from the network.

## S1 pre-change baseline (2026-07-16 UTC)

- **Command**: `/usr/bin/python -m pytest`
- **Result**: passed (11/11 in 0.17s)
- **Source state**: untouched S0 implementation; S0 published artifacts/results unchanged
- **Preflight**: Torch 2.9.1+cu129; CUDA available; NVIDIA A800-SXM4-80GB; ruff unavailable

## S1 red evidence: Torch leaf seam

- **Command**: `/usr/bin/python -m pytest tests/executor/test_torch_parity.py -q`
- **Result**: failed (0/1)
- **Failure**: `ModuleNotFoundError: No module named 'mirage_kan.executor'`
- **Meaning**: the public Torch executor parity test existed and failed before implementation

## Iteration 1 (Phase 1/2: public Torch seam)

- **Score**: 0.85 (lint=skipped format=skipped test=1.00 self=0.75)
- **Lint**: skipped (`ruff` not installed)
- **Format**: skipped (`ruff` not installed)
- **Tests**: passed (12/12)
- **Changes**: `src/mirage_kan/executor/__init__.py`, `src/mirage_kan/executor/torch.py`, `tests/executor/test_torch_parity.py`
- **Feedback**: Public CPU seam preserves exact panel index and keeps expression support independent from membership. Only leaf execution is implemented; operator parity remains.
- **Next**: continue

## Iteration 2 (Phase 1/2: public Torch seam)

- **Score**: 0.89 (lint=skipped format=skipped test=1.00 self=0.82)
- **Lint**: skipped (`ruff` not installed)
- **Format**: skipped (`ruff` not installed)
- **Tests**: passed (13/13)
- **Changes**: `src/mirage_kan/executor/torch.py`, `tests/executor/test_torch_parity.py`
- **Feedback**: Real float64 Torch recursion now covers `Add`, `Sub`, and `SafeDiv`; independently worked values and Pandas support/value parity pass.
- **Next**: next_phase

## Iteration 3 (Phase 2/2: complete scientific parity)

- **Score**: 0.60 (lint=skipped format=skipped test=0.92 self=0.75; test-failure cap applied)
- **Lint**: skipped (`ruff` not installed)
- **Format**: skipped (`ruff` not installed)
- **Tests**: failed (12/13)
- **Changes**: `src/mirage_kan/executor/torch.py`, `tests/executor/test_torch_parity.py`
- **Feedback**: Temporal values and support reached the exact comparison, which exposed a result metadata mismatch: Torch returned `name=None` while Pandas preserved `name="close"` for unary operators.
- **Next**: continue

## Iteration 4 (Phase 2/2: complete scientific parity)

- **Score**: 0.60 (lint=skipped format=skipped test=0.92 self=0.72; test-failure cap applied)
- **Lint**: skipped (`ruff` not installed)
- **Format**: skipped (`ruff` not installed)
- **Tests**: failed (12/13)
- **Changes**: `src/mirage_kan/executor/torch.py`, `tests/executor/test_torch_parity.py`
- **Feedback**: Series-name parity was fixed. The independently worked temporal literal was wrong: A day 4 uses day 2 (`13-11`), and A day 5 is unsupported because day 3 is missing. Torch and Pandas agreed against the mistaken literal.
- **Next**: continue

## Iteration 5 (Phase 2/3: causal temporal parity)

- **Score**: 0.91 (lint=skipped format=skipped test=1.00 self=0.85)
- **Lint**: skipped (`ruff` not installed)
- **Format**: skipped (`ruff` not installed)
- **Tests**: passed (14/14)
- **Changes**: `src/mirage_kan/executor/torch.py`, `tests/executor/test_torch_parity.py`
- **Feedback**: `Delay`, `Delta`, and `Return` match independently corrected literals and exact Pandas Series values/support at causal boundaries. Phase 2 is complete; remaining rolling/cross-sectional/final checks are split into Phase 3.
- **Next**: next_phase

## Iteration 6 (Phase 3/3: rolling, axis, and final verification)

- **Score**: 0.92 (lint=skipped format=skipped test=1.00 self=0.87)
- **Lint**: skipped (`ruff` not installed)
- **Format**: skipped (`ruff` not installed)
- **Tests**: passed (15/15)
- **Changes**: `src/mirage_kan/executor/torch.py`, `tests/executor/test_torch_parity.py`
- **Feedback**: Torch `TsMean` matches worked window-edge literals and exact Pandas values/support with complete-window missing propagation.
- **Next**: continue

## Iteration 7 (Phase 3/3: rolling, axis, and final verification)

- **Score**: 0.60 (lint=skipped format=skipped test=0.93 self=0.78; test-failure cap applied)
- **Lint**: skipped (`ruff` not installed)
- **Format**: skipped (`ruff` not installed)
- **Tests**: failed (14/15)
- **Changes**: `src/mirage_kan/executor/torch.py`, `tests/executor/test_torch_parity.py`
- **Feedback**: `CSRank` reached Torch assignment but integer rank arithmetic defaulted to float32, which cannot be assigned into the float64 result tensor. Cast rank counts to the child value dtype.
- **Next**: continue

## Iteration 8 (Phase 3/3: rolling, axis, and final verification)

- **Score**: 0.97 (lint=skipped format=skipped test=1.00 self=0.95)
- **Lint**: skipped (`ruff` executable not installed; project-local config/dev extra added)
- **Format**: skipped (`ruff` executable not installed; no network installation attempted)
- **Tests**: passed (16/16 in 1.99s, including CUDA parity because CUDA was available)
- **Changes**: `pyproject.toml`, `src/mirage_kan/executor/__init__.py`, `src/mirage_kan/executor/torch.py`, `tests/executor/test_torch_parity.py`, `artifacts/iteration_log.md`
- **Feedback**: All currently supported AST operators execute as real float64 Torch tensor operations and match exact Pandas Series values/support. Membership stays outside expression support; causal lag/rolling edges, CSRank datetime axis, canonical round-trip/identity, CPU, and available CUDA are covered. `/usr/bin/python -m compileall -q src tests` passed. S0 published artifacts/results were not edited or republished.
- **Next**: done

### S1 immutable S0 artifact check

- `factor_libraries/seed_ast_v1/manifest.json`: `6afd41c02302f3d989dfc328a42d8b3e493f157dcd02bd2bd65dbaaed74f7427`
- `factor_libraries/seed_ast_v1/factor_panel.parquet`: `3a551c60c330abd730e44a70cc26f071e9805bc7b5cb5c8f611e931584353fd9`
- `factor_libraries/seed_ast_v1/expression_support.parquet`: `9c0d8cf4cba75f7aefa8655247413d0377c2c5ae2fdf58bb84a7e06ddeae9cb4`
- These remain S0 artifacts bound to the S0 source tree; they do not bind or claim the S1 working-tree state.

# S1 Gate A Capacity Harness Iteration Log

> Scope is generator plus E1/C6/shared harness only. Frozen E2--E5 scientific
> runs are not executed or inspected. Ruff remains unavailable.

## Iteration 1 (Phase 1/4: scalar typed-AST parity)

- **Score**: 0.60 (lint=skipped format=skipped test=0.94 self=0.78; test-failure cap applied)
- **Lint**: skipped (`ruff` unavailable)
- **Format**: skipped (`ruff` unavailable)
- **Tests**: failed (16/17)
- **Changes**: `src/mirage_kan/dsl/core.py`, `src/mirage_kan/executor/torch.py`, DSL/executor parity tests
- **Feedback**: Red evidence first showed `Constant` was unknown. The minimal finite scalar contract and Pandas/Torch broadcasting then reached result metadata, exposing that the Torch result-name helper assumed every non-leaf had a child.
- **Next**: continue

## Iteration 2 (Phase 1/4: scalar typed-AST parity)

- **Score**: 0.91 (lint=skipped format=skipped test=1.00 self=0.85)
- **Lint**: skipped (`ruff` unavailable)
- **Format**: skipped (`ruff` unavailable)
- **Tests**: passed (18/18)
- **Changes**: `src/mirage_kan/executor/torch.py`
- **Feedback**: Scalar constants now have exact finite-real typing, canonical serialization, all-row expression support, and float64 Pandas/Torch parity. Membership remains a separate panel mask.
- **Next**: next_phase

## Iteration 3 (Phase 2/4: exact synthetic panel generator)

- **Score**: 0.92 (lint=skipped format=skipped test=1.00 self=0.87)
- **Lint**: skipped (`ruff` unavailable)
- **Format**: skipped (`ruff` unavailable)
- **Tests**: passed (21/21)
- **Changes**: `src/mirage_kan/experiments/gate_a/data.py`, package exports, `tests/experiments/test_gate_a_data.py`
- **Feedback**: PCG64 split offsets and seven-array date-major draw order reproduce an independently worked OHLCV literal. All candidates run through public AstNode/evaluate_torch, common support is enforced, train median/IQR is reused unchanged, and replay/content hashes are stable.
- **Next**: next_phase

## Iteration 4 (Phase 3/4: E1 and C6 capacity models)

- **Score**: 0.60 (lint=skipped format=skipped test=0.75 self=0.82; test-failure cap applied)
- **Lint**: skipped (`ruff` unavailable)
- **Format**: skipped (`ruff` unavailable)
- **Tests**: failed (3/4 phase tests)
- **Changes**: prospective machine-readable capacity spec, `src/mirage_kan/experiments/gate_a/models.py`, model tests
- **Feedback**: The Cox--de Boor worked literal, exact C6 parameter match, and CUDA float64 test pass. The additive-edge literal reached exact values but its test expected float32 while the required model output is correctly float64.
- **Next**: continue

## Iteration 5 (Phase 3/4: E1 and C6 capacity models)

- **Score**: 0.94 (lint=skipped format=skipped test=1.00 self=0.90)
- **Lint**: skipped (`ruff` unavailable)
- **Format**: skipped (`ruff` unavailable)
- **Tests**: passed (25/25)
- **Changes**: corrected float64 worked-test literal only
- **Feedback**: E1 is a real additive cubic B-spline KAN with edge outputs/curves and 115 declared parameters. The prospectively frozen E3/E4 spec declares 361 E4 parameters; C6 is a genuine one-hidden-layer SiLU MLP with width 45 and exactly 361 parameters. CPU/CUDA float64 pass.
- **Next**: next_phase

## Iteration 6 (Phase 4/4: shared train/checkpoint/test-once seam)

- **Score**: 0.94 (lint=skipped format=skipped test=1.00 self=0.90)
- **Lint**: skipped (`ruff` unavailable)
- **Format**: skipped (`ruff` unavailable)
- **Tests**: passed (26/26)
- **Changes**: `src/mirage_kan/experiments/gate_a/training.py`, training behavior test, package exports
- **Feedback**: The generic neural-arm seam uses AdamW/float64, training noisy labels, validation-clean checkpoint selection, atomic categorized checkpoints/manifests, exact step/sample/parameter/wall/peak-memory accounting, and exclusive test-once claims. Console output is mirrored to the manifest-linked raw log.
- **Next**: continue

## Iteration 7 (Phase 4/4: persisted inputs and runnable smoke entry point)

- **Score**: 0.95 (lint=skipped format=skipped test=1.00 self=0.92)
- **Lint**: skipped (`ruff` unavailable)
- **Format**: skipped (`ruff` unavailable)
- **Tests**: passed (27/27)
- **Changes**: generated-array persistence and hash manifest, required pre-training data-manifest link, tiny smoke module, persistence behavior test
- **Feedback**: All returned raw panels, indices, candidate arrays, clean/noisy targets, masks, and scaler statistics are persisted before training. The run manifest cross-links the immutable generated-data manifest and console log.
- **Next**: continue

## Iteration 8 (Phase 4/4: smoke command invocation)

- **Score**: 0.60 (lint=skipped format=skipped test=1.00 self=0.70; runnable-smoke failure cap applied)
- **Lint**: skipped (`ruff` unavailable)
- **Format**: skipped (`ruff` unavailable)
- **Tests**: previously passed (27/27)
- **Changes**: no source changes; failed command output preserved in `logs/s1_gate_a_e1_tiny_cpu/console.log`
- **Feedback**: Direct module invocation failed before data generation because this source-layout project is not installed and the CLI command omitted `PYTHONPATH=src`. The entry point itself was not reached.
- **Next**: continue with the explicit source path

## Iteration 9 (Phase 4/4: reproducible CPU/CUDA harness smoke)

- **Score**: 0.96 (lint=skipped format=skipped test=1.00 self=0.94)
- **Lint**: skipped (`ruff` unavailable)
- **Format**: skipped (`ruff` unavailable)
- **Tests**: passed (27/27)
- **Changes**: deterministic C6 initialization, selected-checkpoint reload before test-once evaluation, explicit resolved/source config hash provenance
- **Feedback**: Tiny E1 CPU and seeded C6 CUDA commands completed through generated-data persistence, validation checkpointing, accounting, test-once enforcement, and manifest-linked raw console logs. The smokes are intentionally outside the frozen seeds/full panel and are not scientific Gate A reads.
- **Next**: final verification

## Iteration 10 (Phase 4/4: final verification)

- **Score**: 0.97 (lint=skipped format=skipped test=1.00 self=0.95)
- **Lint**: skipped honestly (`ruff` executable unavailable; no dependency installed)
- **Format**: skipped honestly (`ruff` executable unavailable)
- **Tests**: passed (27/27 in 3.46s)
- **Changes**: final verification and cache cleanup only
- **Feedback**: `compileall` passed; both seal hashes and the lock hash are unchanged; the official pykan reference remains clean at the pinned commit with matching source hashes; all three S0 published artifact hashes are unchanged; generated Python caches were removed. No frozen full scientific matrix was run or inspected.
- **Next**: done

# S1 Gate A Symbolic Models Iteration Log

> Scope is E2--E4 implementation and train/validation-only tiny integration.
> The frozen three-seed scientific matrix and E5 remain untouched. Ruff is not
> installed, so lint/format checks are recorded as unavailable.

## Symbolic pre-change baseline (2026-07-16 UTC)

- **Command**: `PYTHONPATH=src python -m pytest -q`
- **Result**: passed (27/27 in 3.37s)
- **Preflight**: Torch 2.9.1+cu129, CUDA 12.9 available, two NVIDIA A800-SXM4-80GB GPUs
- **Seal hashes**: protocol `ece291...4c1ed`, config `fd3da6...aec9`, lock `dea525...b6b88`

## Symbolic red evidence: primitives and hard interface

- **Command**: `PYTHONPATH=src python -m pytest tests/experiments/test_gate_a_symbolic.py -q`
- **Result**: failed (0/4)
- **Failure**: missing `gate_a.symbolic` module and prospective symbolic spec
- **Meaning**: frozen primitive execution, independent hard export, fidelity, residual-energy semantics, and prospective choices were tested before implementation

## Iteration 1 (Phase 1/4: primitive and hard symbolic interfaces)

- **Score**: 0.92 (lint=skipped format=skipped test=1.00 self=0.86)
- **Lint**: skipped (`ruff` unavailable)
- **Format**: skipped (`ruff` unavailable)
- **Tests**: passed (31/31)
- **Changes**: `configs/model_specs/s1_gate_a_symbolic_v0.json`, `src/mirage_kan/experiments/gate_a/symbolic.py`, `tests/experiments/test_gate_a_symbolic.py`
- **Feedback**: all eight frozen primitives have literal finite-domain Torch tests; hard models retain only selected primitives and affine buffers; canonical serialization, shape/fidelity metadata, explicit zero-energy handling, and prospective promotion-family governance are fixed before smoke metrics.
- **Next**: next_phase

## Symbolic red evidence: E3/E4 train-time models

- **Command**: `PYTHONPATH=src python -m pytest tests/experiments/test_gate_a_symbolic.py -q`
- **Result**: failed (4/7 passed)
- **Failure**: missing public `SymbolicKAN` and `SymbolicResidualKAN` classes
- **Meaning**: real train-time gates, deterministic hardening, exact E4 capacity, real cubic residuals, residual energy, and 801-point shape outputs were specified before implementation

## Iteration 2 (Phase 2/4: E3/E4 train-time models)

- **Score**: 0.93 (lint=skipped format=skipped test=1.00 self=0.88)
- **Lint**: skipped (`ruff` unavailable)
- **Format**: skipped (`ruff` unavailable)
- **Tests**: passed (34/34)
- **Changes**: `src/mirage_kan/experiments/gate_a/models.py`, `src/mirage_kan/experiments/gate_a/symbolic.py`, `tests/experiments/test_gate_a_symbolic.py`
- **Feedback**: E3 has per-edge eight-way soft gates, four affine parameters per primitive, scheduled temperature, entropy/sparsity accounting, and stable dictionary-order hardening. E4 reuses that exact analytical path and adds only 19 cubic coefficients plus one residual scale per edge, reaching the frozen 361 parameters. Non-finite primitive inputs now fail explicitly rather than being silently imputed.
- **Next**: next_phase

## Symbolic red evidence: E2 post-hoc fitting

- **Command**: `PYTHONPATH=src python -m pytest tests/experiments/test_gate_a_symbolic.py -q`
- **Result**: failed (7/9 passed)
- **Failure**: missing `gate_a.posthoc` module
- **Meaning**: exact eight-fits-per-edge budgeting, target-free API, selected-checkpoint loading, E1 bias preservation, and independent hard execution were tested before implementation

## Iteration 3 (Phase 3/4: E2 selected-E1 symbolification)

- **Score**: 0.94 (lint=skipped format=skipped test=1.00 self=0.90)
- **Lint**: skipped (`ruff` unavailable)
- **Format**: skipped (`ruff` unavailable)
- **Tests**: passed (36/36)
- **Changes**: `src/mirage_kan/experiments/gate_a/posthoc.py`, `src/mirage_kan/experiments/gate_a/symbolic.py`, `tests/experiments/test_gate_a_symbolic.py`
- **Feedback**: E2 loads a selected E1 state without optimization and independently runs SciPy TRF least-squares exactly once for each of eight frozen families on every training edge. Selection uses only train inputs and E1 edge outputs, preserves E1 output bias, records SciPy/settings/budget, and returns an independent hard analytical model. Primitive inputs are neither imputed nor clipped; non-finite input and output paths fail explicitly.
- **Next**: next_phase

# S1 Gate A E5 Typed Symbolic Regression Iteration Log

> Scope is the E5 implementation and one train/validation-only non-scientific
> smoke. The frozen three-seed matrix is not run. Ruff is unavailable, so lint
> and format checks are skipped rather than installing a dependency.

## E5 red evidence: concrete typed AST seam

- **Command**: `PYTHONPATH=src /usr/bin/python -m pytest tests/experiments/test_gate_a_e5.py -q --tb=short`
- **Result**: failed (0/1)
- **Failure**: `ModuleNotFoundError: No module named 'mirage_kan.experiments.gate_a.e5'`
- **Meaning**: the frozen spec and independently executable AST contract preceded implementation

## Iteration 1 (Phase 1/4: frozen grammar and executable AST)

- **Score**: 0.92 (lint=skipped format=skipped test=1.00 self=0.86)
- **Lint**: skipped (`ruff` unavailable)
- **Format**: skipped (`ruff` unavailable)
- **Tests**: passed (44/44)
- **Changes**: `configs/model_specs/s1_gate_a_e5_v0.json`, `src/mirage_kan/experiments/gate_a/e5.py`, `tests/experiments/test_gate_a_e5.py`
- **Feedback**: The E5 grammar, deterministic generation/selection choices, invalid-domain policy, exact ceilings, duplicate policy, and persistence schema were frozen before any E5 metric. Concrete float64 ASTs use exact hexadecimal constants, round-trip independently, reject non-finite values, retain source/window identities, and count two/three-term programs as 10/14 nodes at depth 4. A mistaken worked literal was exposed by the red/green loop and corrected independently.
- **Next**: next_phase

## E5 red evidence: genuine structural-search seam

- **Command**: `PYTHONPATH=src /usr/bin/python -m pytest tests/experiments/test_gate_a_e5.py -q --tb=short`
- **Result**: failed (1/2 passed)
- **Failure**: missing public `E5SearchSettings` and `search_e5`
- **Meaning**: multi-source recovery and a test-free search signature were specified before search implementation

## Iteration 2 (Phase 2/4: deterministic search, fit, selection, and budget)

- **Score**: 0.94 (lint=skipped format=skipped test=1.00 self=0.90)
- **Lint**: skipped (`ruff` unavailable)
- **Format**: skipped (`ruff` unavailable)
- **Tests**: passed (47/47)
- **Changes**: `src/mirage_kan/experiments/gate_a/e5.py`, `tests/experiments/test_gate_a_e5.py`, prospective E5 accounting fields
- **Feedback**: The deterministic single/pair/beam-triple search recovers an independently worked two-source additive formula with train-only least squares and validation-clean selection. A 0.005 NRMSE equivalence set selects the lower executable complexity. Duplicate, invalid-AST, invalid-execution, fit, and budget-exhaustion accounting are disjoint; duplicates do not consume the exact valid full-fit ceiling, and deterministic reruns preserve ledger and model bytes.
- **Next**: next_phase

## E5 red evidence: immutable export seam

- **Command**: `PYTHONPATH=src /usr/bin/python -m pytest tests/experiments/test_gate_a_e5.py -q --tb=short`
- **Result**: failed (4/5 passed)
- **Failure**: missing public `load_e5_export` and `save_e5_search`
- **Meaning**: source-mass semantics, reconstructability, integrity checks, manifest-last publication, and no-replace behavior preceded persistence implementation

## Iteration 3 (Phase 3/4: source attribution and immutable persistence)

- **Score**: 0.95 (lint=skipped format=skipped test=1.00 self=0.92)
- **Lint**: skipped (`ruff` unavailable)
- **Format**: skipped (`ruff` unavailable)
- **Tests**: passed (49/49)
- **Changes**: E5 immutable coefficient buffers, centered source-energy attribution, hash-verified ledger/model/spec export and loader, persistence behavior tests
- **Feedback**: Source mass is invariant to additive feature shifts for affine terms, sums to one when defined, and reports an explicit all-zero state. Ledger and selected model are fsynced before an exclusively published final manifest; all payload and prospective-spec hashes are checked on load. Missing referenced columns, empty samples, non-finite inputs, and overflowing primitive outputs fail explicitly without imputation.
- **Next**: next_phase

## E5 red evidence: categorized smoke seam

- **Command**: `PYTHONPATH=src /usr/bin/python -m pytest tests/experiments/test_gate_a_e5.py -q --tb=short`
- **Result**: failed (5/6 passed)
- **Failure**: missing `gate_a.e5_smoke` module
- **Meaning**: fresh-seed enforcement, reduced budget, train/validation-only access, and categorized smoke publication were specified before the runnable entry point

## Iteration 4 (Phase 4/4: train/validation-only smoke and final verification)

- **Score**: 0.97 (lint=skipped format=skipped test=1.00 self=0.95)
- **Lint**: skipped honestly (`ruff` executable unavailable; no package installed)
- **Format**: skipped honestly (`ruff` executable unavailable)
- **Tests**: passed (49/49; E5-focused deterministic rerun 6/6)
- **Changes**: `src/mirage_kan/experiments/gate_a/e5_smoke.py`, minimal package exports, categorized non-scientific smoke artifacts
- **Feedback**: Fresh seed 8675309 and reduced 96-candidate budget completed with 96 distinct successful fits plus one recorded budget-exhaustion attempt. A final audit grouped same-source term contributions before energy attribution and added explicit automatic-vs-audit generation mode; the original smoke is preserved and the current `s1_gate_a_e5_tiny_cpu_v2` manifest records the frozen automatic generator. Both manifests record only train and validation access and `test_evaluated=false`; no test metric is produced. `compileall`, deterministic replay, full pytest, sealed prereg/config hashes, and all three S0 publication hashes pass unchanged.
- **Next**: done

## Symbolic red evidence: trainer wiring and immutable publication

- **Command**: `PYTHONPATH=src python -m pytest tests/experiments/test_gate_a_data.py tests/experiments/test_gate_a_training.py tests/experiments/test_gate_a_symbolic.py -q`
- **Result**: failed (13/17 passed)
- **Failure**: data publication replaced an existing directory, trainer did not call/persist symbolic penalties or reserve log namespace, and hard export was absent
- **Meaning**: no-replace publication, actual objective wiring, per-step/selected penalty evidence, and reconstructable hard checkpoints were tested before implementation

## Iteration 4 (Phase 4/4: trainer, publication, and smoke seams)

- **Score**: 0.95 (lint=skipped format=skipped test=1.00 self=0.92)
- **Lint**: skipped (`ruff` unavailable)
- **Format**: skipped (`ruff` unavailable)
- **Tests**: passed (39/39)
- **Changes**: `src/mirage_kan/experiments/gate_a/data.py`, `training.py`, `symbolic.py`, `symbolic_smoke.py`, package exports, data/training/symbolic tests
- **Feedback**: generated data and every run leaf namespace are reserved no-replace and published manifest-last. The shared trainer now invokes symbolic penalties before every prediction, logs penalty totals, and persists full step plus selected-checkpoint temperature/entropy/sparsity/residual accounting while E1/C6 remain penalty-free. Hard exports are hash-bound, canonical, independently reconstructable, and manifest-last. A separate E2/E3/E4 smoke entry point never calls test evaluation.
- **Next**: final_verification

## Iteration 5 (Phase 4/4: selected-state fidelity and tiny CPU/CUDA integration)

- **Score**: 0.96 (lint=skipped format=skipped test=1.00 self=0.94)
- **Lint**: skipped (`ruff` unavailable)
- **Format**: skipped (`ruff` unavailable)
- **Tests**: passed (40/40)
- **Changes**: selected gate-temperature checkpoint/restore wiring, exact epsilon-stabilized residual-energy semantics, source-mass metadata, exclusive metric JSON publication, hard-export reload test
- **Feedback**: a regression test forces the best symbolic checkpoint to occur before the final optimizer step and verifies its inference temperature is restored in memory and persisted in checkpoint/manifest. CPU E2/E3 and CUDA E4 train/validation-only smokes completed; E3/E4 both selected step 1 and retained temperature 1.05 after a step-2 anneal. Every source training manifest still records `test_once.evaluated=false`.
- **Smoke commands**: `PYTHONPATH=src python -m mirage_kan.experiments.gate_a.symbolic_smoke --arm E2 --run-id s1_gate_a_e2_tiny_cpu --device cpu --steps 2`, corresponding E3 CPU command, and E4 `--device cuda:0`
- **Outputs**: `artifacts/s1_gate_a_symbolic_smoke/{data,checkpoints,manifests,symbolic_exports}` and `logs/s1_gate_a_{e2_tiny_cpu,e3_tiny_cpu,e4_tiny_cuda}/console.log`
- **Next**: final_verification

## Symbolic final verification (2026-07-16 UTC)

- **Result**: `PYTHONPATH=src python -m compileall -q src tests` passed; full pytest passed (40/40 in 4.47s)
- **Environment**: CPU paths and real CUDA float64 E4 path passed on NVIDIA A800-SXM4-80GB; ruff remained unavailable and was not installed
- **Seals**: prereg protocol/config hashes match their lock; lock hash remains `dea525...b6b88`; frozen capacity spec remains unchanged at `5f4497...6195`
- **S0 artifacts**: manifest `6afd41...f7427`, factor panel `3a551c...3fd9`, expression support `9c0d8c...fd9` unchanged
- **Evidence boundary**: no E5 code, frozen three-seed matrix, test-once metric, promotion fit, or scientific Gate A conclusion was produced
- **Final score**: 0.96 (tests/compile/smokes/seals pass; lint/format unavailable)

# S1 Symbolic Acceptance Follow-up

## Follow-up pre-change baseline

- **Command**: `PYTHONPATH=src python -m pytest -q`
- **Result**: passed (40/40 in 4.34s)
- **Scope**: correct residual penalty semantics, source-recovery bias invariance, manifest-last ordering, and runtime dependency ledger only

## Follow-up red evidence

- **Residual penalty/source mass**: targeted symbolic tests failed because accounting exposed only raw residual MSE and source mass changed after adding constant edge biases.
- **Manifest ordering**: an explicit publication-event test failed with final event `log`, proving the training manifest preceded its final console append; the test-once path had the analogous ordering.
- **Dependency ledger**: `test_gate_a_runtime_dependencies_are_declared` failed because neither SciPy nor Torch was declared in `project.dependencies`.

## Iteration 6 (acceptance corrections)

- **Score**: 0.96 (lint=skipped format=skipped test=1.00 self=0.94)
- **Lint**: skipped (`ruff` unavailable)
- **Format**: skipped (`ruff` unavailable)
- **Tests**: passed (43/43 in 4.52s)
- **Changes**: E4 now penalizes the mean per-edge proposal rho while separately accounting raw scaled-residual energy and rho; E3 reports both as zero. E2/E3/E4 source mass uses mean-centered edge outputs and returns an explicit no-selection record when all contribution energy is zero. Training/test-once logs and metrics now finish before atomic manifest publication/update. SciPy and Torch runtime requirements are declared without installation.
- **Evidence boundary**: no sealed prereg/config edit, scientific-matrix run, E5 change, or new metric-based model choice.
- **Next**: final_verification

## Follow-up final verification (2026-07-16 UTC)

- `PYTHONPATH=src python -m compileall -q src tests`: passed
- Full pytest: passed (43/43 in 4.40s)
- CUDA float64 E4 objective/forward: passed with 361 parameters and both `spline_residual_raw_energy` and `spline_residual_ratio` accounting fields
- Sealed protocol/config/lock and all three S0 artifact hashes: unchanged
- Ruff: unavailable; no package installation attempted
- **Final follow-up score**: 0.96

# S1 Gate A unified scientific harness (More Effort)

## Iteration 1 (Phase 1/4: prospective controls and promotion governance)

- **Score**: 0.94 (lint=skipped format=skipped test=1.00 self=0.90)
- **Lint**: skipped (`ruff` executable unavailable; no installation attempted)
- **Format**: skipped (`ruff` executable unavailable)
- **Red evidence**: new controls tests failed 2/2 and promotion tests failed 4/4 with missing public modules before implementation
- **Tests**: passed (controls 2/2; promotion 4/4 under `-W error`)
- **Changes**: prospectively froze promotion and matrix-runner choices; implemented exact whole-date null mapping, exact source removal, sign-preserving cross-seed eligibility, three bounded continuous promotion families, absolute low-complexity fit gates, governance audits, and continuous-only independently reloadable hard refit
- **Feedback**: initial green exposed NumPy 2 `ptp`, eager inactive rational branches, and readonly-array Torch conversion; each root cause was fixed, then all promotion tests passed with warnings promoted to errors. No promotion or matrix smoke metric was observed.
- **Next**: next_phase

## Iteration 2 (Phase 2/4: evaluator metrics, bootstrap, and seven literal gates)

- **Score**: 0.95 (lint=skipped format=skipped test=1.00 self=0.92)
- **Lint/format**: skipped (`ruff` executable unavailable)
- **Red evidence**: four evaluator tests failed before `metrics.py` existed
- **Tests**: passed (promotion + metrics 9/9 under `-W error`)
- **Changes**: added the worked raw-Return5-to-standardized shape coordinate, evaluator-only clean mechanism, deterministic circular moving-date-block percentile bootstrap, one exclusive all-arm test opening with row-aligned hashed arrays, explicit N/A semantics, and literal conditions 1–7 with capacity-inconclusive status
- **Feedback**: failure injection proves a predictor exception preserves the exclusive claim but publishes no final opening manifest and marks no individual arm evaluated. Residual arrays are defensive readonly copies; any non-identical frozen z grid fails closed; continuity now reads the prospective tolerance rather than a code literal.
- **Next**: next_phase

## Iteration 3 (Phase 3/4: exact unified matrix orchestration)

- **Score**: 0.95 (lint=skipped format=skipped test=1.00 self=0.92)
- **Lint/format**: skipped (`ruff` executable unavailable)
- **Red evidence**: matrix tests failed 2/2 before `matrix.py` existed; first green attempt then exposed a cross-arm training-log namespace collision
- **Tests**: passed (matrix 2/2; combined harness 14/14 under `-W error`)
- **Changes**: added scientific/smoke mode validation, immutable sealed-config comparison, real E1–E5+C6 orchestration, per-seed null/source-removed E4 controls, frozen residual publication, governed cross-seed promotion, auxiliary E3/E4 hard audit predictions, and one exclusive all-predictor test opening
- **Feedback**: scientific mode rejects API overrides and post-construction config mutation before root creation. Each non-training arm receives a preflightable test manifest. Null and source-removed controls now use separate pre-training input manifests and post-training final evidence manifests. Neural/E2/E5/promotion FLOPs follow the prospective estimators.
- **Next**: next_phase

## Iteration 4 (Phase 4/4: fresh-seed smoke and final verification)

- **Score**: 0.97 (lint=skipped format=skipped test=1.00 self=0.95)
- **Lint/format**: skipped (`ruff` is not installed; no installation attempted)
- **Tests**: full suite passed (63/63 under `-W error`); `compileall` passed
- **Smoke command**: `PYTHONPATH=src /usr/bin/python -c 'from mirage_kan.experiments.gate_a.matrix import smoke_matrix_settings, run_gate_a_matrix; s=smoke_matrix_settings("s1_gate_a_matrix_fresh_cpu_v1", seeds=(8675401,8675402), assets=2, burn_in_dates=25, train_dates=40, validation_dates=40, test_dates=40, max_steps=1, batch_size=16, e5_candidate_budget=2, bootstrap_replicates=2); print(run_gate_a_matrix(s, artifact_base="artifacts", device="cpu"))'`
- **Smoke output**: `artifacts/s1_gate_a_matrix_smoke/s1_gate_a_matrix_fresh_cpu_v1/`; both fresh seeds materialized E1/E2/E3/E4/E5/C6 plus E3/E4 audit predictions in one opening, null promotions were zero, source-removed evidence was final-manifest-last, and proposal/prereg/config seals were unchanged. `promotion_status=ineligible_residual_shapes` is expected non-scientific one-step smoke evidence, not a Gate A result.
- **CUDA smoke**: real E4 float64 forward/objective/backward plus independent promoted-hard CUDA execution passed on NVIDIA A800-SXM4-80GB with finite outputs.
- **Evidence boundary**: no frozen seed (1729/2718/31415) was trained or test-opened; no scientific Gate A or Alpha-profitability claim was made.
- **Next**: done

## Follow-up: four-affine audit and prospective v1 supersession

- **Finding**: the complete initial-dictionary duplicate audit must fit input scale, input bias, output scale, and output bias. Under that exact audit, the v0 clean mechanism was approximated by affine `Tanh` with Pearson `0.9990530523` and NRMSE `0.0435086048` on `[-6,6]`, which is inside the prospectively frozen duplicate NRMSE threshold `0.05`.
- **Decision**: because no frozen seed had been trained or test-opened, v1 prospectively supersedes v0 and changes only the synthetic mechanism constants. The original v0 protocol, config, and lock remain unchanged historical records. The scientific seeds, arms, budgets, controls, gates, and evaluation rules are unchanged.
- **Active seals**: proposal `1880ccf...c41a`; v1 protocol `a91f6b97...ab97f`; v1 config `3af2a7f4...c11a`; v1 lock `6be7a167...f4e4d`.
- **Audit behavior**: candidate-family approximate duplicates use the frozen Pearson/NRMSE gate; exact scaled copies of existing residual dictionary primitives are independently rejected at numerical-identity tolerance. Audit FLOPs are included in promotion accounting.
- **Evidence boundary**: this correction used source/code audit and reduced fresh-seed smoke only. No seed in `{1729, 2718, 31415}` was trained, selected, promoted, or test-opened.

## V1 full-chain smoke and final verification (2026-07-16 UTC)

- **Smoke command**: `PYTHONPATH=src /usr/bin/python -c 'from mirage_kan.experiments.gate_a.matrix import smoke_matrix_settings, run_gate_a_matrix; s=smoke_matrix_settings("s1_gate_a_v1_matrix_fresh_cpu_v1", seeds=(8675501,8675502), assets=2, burn_in_dates=25, train_dates=40, validation_dates=40, test_dates=40, max_steps=1, batch_size=16, e5_candidate_budget=2, bootstrap_replicates=2); print(run_gate_a_matrix(s, artifact_base="artifacts", device="cpu"))'`
- **Output**: `artifacts/s1_gate_a_matrix_smoke/s1_gate_a_v1_matrix_fresh_cpu_v1/`; both fresh seeds completed E1/E2/E3/E4/E5/C6, both controls, E3/E4 hard audits, and exactly one aligned test opening per seed. Proposal/protocol/config and implementation snapshots were unchanged across execution.
- **Expected smoke result**: `promotion_status=ineligible_residual_shapes` after one optimizer step is connectivity evidence only, not a Gate A result. Null promotions were zero; JSON contained no NaN or Infinity.
- **Tests**: `PYTHONPATH=src /usr/bin/python -m compileall -q src tests` passed; `/usr/bin/python -W error -m pytest -q` passed 63/63 in 48.28 seconds.
- **CUDA**: the real float64 E4 objective/forward/backward and independently executable promoted-hard path passed on NVIDIA A800-SXM4-80GB with finite outputs.
- **Lint/format**: skipped because `ruff` is unavailable; no dependency was installed.
- **Scientific boundary**: the formal three-seed run remains intentionally unexecuted. No Gate A or alpha-profitability claim is made.

## Final editorial-seal smoke (2026-07-16 UTC)

- **Reason**: removed one unused type import and restored LaTeX around the asset index in the v1 protocol. The protocol lock was regenerated, so a new fresh-seed smoke was run to bind connectivity evidence to the final source and seal identities.
- **Command**: `PYTHONPATH=src /usr/bin/python -c 'from mirage_kan.experiments.gate_a.matrix import smoke_matrix_settings, run_gate_a_matrix; s=smoke_matrix_settings("s1_gate_a_v1_matrix_final_cpu_v1", seeds=(8675601,8675602), assets=2, burn_in_dates=25, train_dates=40, validation_dates=40, test_dates=40, max_steps=1, batch_size=16, e5_candidate_budget=2, bootstrap_replicates=2); print(run_gate_a_matrix(s, artifact_base="artifacts", device="cpu"))'`
- **Output**: `artifacts/s1_gate_a_matrix_smoke/s1_gate_a_v1_matrix_final_cpu_v1/`; `seal_unchanged=true`, `implementation_unchanged=true`, implementation aggregate `0e346b1c...e0bf9`.
- **Opening audit**: both seeds contain exactly `C6`, `E1`, `E2`, `E3`, `E3_HARD`, `E4`, `E4_HARD_ANALYTICAL`, and `E5` in one aligned test opening. Null promotions are zero; JSON non-finite scan is clean.
- **Expected one-step status**: `ineligible_residual_shapes`; this remains connectivity evidence only.
- **Final tests**: compileall passed; pytest passed 63/63 under `-W error` in 47.51 seconds.
- **Final seals**: protocol `a91f6b97ba6d7c172cc91b7313a83c069bdd79e88e04fc03dd2c2a960fcab97f`; lock `6be7a1676e79486f3b8b759f6363bd772d2352eca0a8b10babc6b41f808f4e4d`.
- **Evidence boundary**: no frozen scientific seed was used and no scientific conclusion was produced.

# S1 Gate A independent-audit corrections (More Effort)

## Iteration 1 (Phase 1/5: immutable opening inputs and checkpoint identity)

- **Score**: 0.91 (lint=skipped format=skipped test=1.00 self=0.85)
- **Lint/format**: skipped because `ruff` is unavailable; no package installation attempted
- **Red evidence**: four targeted tests failed: successful opening rewrote input manifests, failed predictors had no per-arm claim, partial claim collisions did not fail closed, and training manifests omitted selected-checkpoint hashes.
- **Green**: 4/4 targeted tests passed under `-W error`.
- **Changes**: per-arm sibling claims, immutable preflight path/hash records, immutable per-arm receipts, byte-identical input manifest enforcement, predictor-failure fail-closed behavior, selected-checkpoint SHA-256 publication and pre-load verification.
- **Next**: next_phase

## Iteration 2 (Phase 2/5: complete arm schema and aggregation)

- **Score**: 0.92 (lint=skipped format=skipped test=1.00 self=0.87)
- **Lint/format**: skipped because `ruff` is unavailable
- **Red evidence**: schema constants and full cross-seed aggregation were absent; the integration report omitted multiple required metrics and used unreasoned raw `N/A` strings.
- **Green**: schema/aggregation plus full-chain integration tests passed 2/2 under `-W error`; promoted-hard source reporting test also passed.
- **Changes**: registered 20 canonical numeric metrics for all E1/E2/E3/E4/E5/C6/HARD arms; every missing value is `{value: "N/A", reason: ...}`; flattened source recovery, fidelity, residual, AST complexity, budget, FLOPs, time, and memory evidence; added per-metric median/full range/available/missing seed aggregation; HARD-absent reports fill every field independently.
- **Next**: next_phase

## Iteration 3 (Phase 3/5: pretest freeze and terminal transaction)

- **Score**: 0.93 (lint=skipped format=skipped test=1.00 self=0.88)
- **Lint/format**: skipped because `ruff` is unavailable
- **Red evidence**: the injected post-open fault test failed because the runner had no transaction hook or terminal-failure contract; train/shape/accounting work still occurred after opening.
- **Green**: injected failure test passed 1/1; success-path matrix plus all metric tests passed 8/8 under `-W error`.
- **Changes**: all shape/source/accounting/complexity evidence and identity checks now publish as immutable pretest summaries plus `pretest_ready.json` before any claim; post-open merging reads only immutable prediction/metric artifacts and frozen pretest evidence; any opening/postprocess exception writes exclusive `terminal_failure.json` with stage, error, opened seeds, claims, and prediction-only recovery evidence, then re-raises without a success manifest.
- **Next**: next_phase

## Iteration 4 (Phase 4/5: scientific implementation lock and artifact graph)

- **Score**: 0.94 (lint=skipped format=skipped test=1.00 self=0.90)
- **Lint/format**: skipped because `ruff` is unavailable
- **Red evidence**: CLI still exposed `--artifact-base`; implementation lock was absent; the top manifest did not provide a hash-verifiable child graph.
- **Green**: CLI/lock tests passed 2/2 and all matrix transaction/index tests passed 4/4 under `-W error`.
- **Changes**: scientific CLI now fixes output under project `artifacts/`; `prereg/s1_gate_a_v1_implementation.lock.json` freezes aggregate and per-file implementation identities plus model/matrix specs while excluding itself; scientific validation fails on any mismatch; top success manifest indexes and hashes data, openings, seed summaries, controls, promotion, reports, event log, and pretest readiness.
- **Next**: next_phase

## Iteration 5 (Phase 5/5: final verification and exact-hash fresh smoke)

- **Score**: 0.96 (lint=skipped format=skipped test=1.00 self=0.94)
- **Lint/format**: skipped because `ruff` is unavailable; no installation attempted
- **Tests**: compileall passed; full pytest passed 66/66 under `-W error` in 52.23 seconds.
- **CUDA**: NVIDIA A800-SXM4-80GB passed real float64 E4 forward/objective/backward and promoted-HARD execution; all outputs/gradients were finite and HARD reported source `Return(Close,5)`.
- **Smoke command**: `PYTHONPATH=src /usr/bin/python -c 'from mirage_kan.experiments.gate_a.matrix import smoke_matrix_settings, run_gate_a_matrix; s=smoke_matrix_settings("s1_gate_a_v1_auditfix_final_cpu_v1", seeds=(8675701,8675702), assets=2, burn_in_dates=25, train_dates=40, validation_dates=40, test_dates=40, max_steps=1, batch_size=16, e5_candidate_budget=2, bootstrap_replicates=2); print(run_gate_a_matrix(s, artifact_base="artifacts", device="cpu"))'`
- **Smoke output**: `artifacts/s1_gate_a_matrix_smoke/s1_gate_a_v1_auditfix_final_cpu_v1/manifests/matrix.json`; proposal/v1/implementation identities and implementation aggregate `5aef31e8...d74f` remained unchanged.
- **Integrity audit**: every core-child path/hash verified; both seeds had eight immutable arm receipts whose input-manifest before/after hashes equal current bytes; every reported arm had all 20 canonical numeric fields; no bare unreasoned `N/A`, NaN, or Infinity occurred; null promotions were zero.
- **Expected status**: `ineligible_residual_shapes` after one optimizer step is connectivity evidence only, never a Gate A result.
- **Locks**: v1 lock `6be7a167...f4e4d`; implementation lock `1f68aab0...54cc`.
- **Evidence boundary**: no seed in `{1729, 2718, 31415}` was run or test-opened; no scientific or profitability claim exists.
- **Next**: done

# S1 Gate A second independent-audit corrections

## Governance incident: unsafe formal-root red test

- **Classification**: `pre-test partial scientific attempt, invalidated`; adjudicated invalid before test opening and eligible for a clean rerun after audit approval.
- **Cause**: a red test invoked a valid scientific runner with a non-default temporary artifact root before the rejection guard existed. It entered seed-1729 data generation and partial E1 training before interruption.
- **Retained evidence**: `/tmp/pytest-of-root/pytest-134/test_matrix_modes_reject_scien0/s1_gate_a_scientific/fixed_project_artifacts`; complete inventory, hashes, timeline, and scope are in `governance/incidents/2026-07-16_frozen_seed_red_test_incident.md`.
- **Boundary**: generated train/validation/test arrays, one E1 checkpoint, event log, and console validation/training log exist. No training manifest, test claim, opening, prediction, receipt, metric, Gate report, terminal manifest, or success manifest exists.
- **State**: project `artifacts/` contains no formal scientific run. Test-once entitlement was not consumed. The v1 scientific-results state was not edited. Subsequent design changes use only the audit blockers, never retained data/checkpoint/log contents.
- **Correction**: scientific artifact-root rejection now precedes seal access, root preparation, generator, and trainer; sentinel tests prove those seams are never called.

## Iteration 6 (Phase 1/4: path and formal-root confinement)

- **Score**: 0.93 (lint=skipped format=skipped test=1.00 self=0.88)
- **Tests**: 9/9 targeted guard, safe-run-ID, truthful-lock, and incident-ledger tests passed under `-W error`.
- **Changes**: run IDs require one safe filename segment; resolved-root containment is checked; programmatic scientific runs reject non-project artifact roots before any side effect.
- **Next**: next_phase

## Iteration 7 (Phase 2/4: complete artifact index)

- **Score**: 0.95 (lint=skipped format=skipped test=1.00 self=0.92)
- **Tests**: fresh-only matrix index integration passed. Exact indexed file-set equality, every path/hash/byte count, canonical aggregate, and tampered training-log detection passed.
- **Changes**: a successful top manifest recursively indexes every other regular file under the run root, including checkpoints, model/training manifests, residual shapes, claims, receipts, arrays, curves, logs, metrics, and reports. `matrix.json` is excluded to avoid self-reference; terminal evidence forbids success publication.
- **Next**: next_phase

## Iteration 8 (Phase 3/4: truthful implementation lock and full regression)

- **Score**: 0.96 (lint=skipped format=skipped test=1.00 self=0.94)
- **Tests**: compileall passed; full pytest passed 74/74 under `-W error` in 52.58 seconds.
- **Lock**: snapshot aggregate `f5675a84...a4d8`; lock records a creation time later than the exact maximum locked-file mtime and includes the hash-bound adjudicated incident ledger.
- **Evidence boundary**: full regression used sentinel-protected formal-root validation and fresh-only matrix fixtures; it did not resume the retained partial attempt.
- **Next**: final_verification

## Iteration 9 (Phase 4/4: CUDA and fresh-smoke custody audit)

- **CUDA**: NVIDIA A800-SXM4-80GB passed E4 float64 forward, regularized objective, backward gradients, analytical hardening, and independently executable promoted-HARD inference; all values were finite and the promoted source was `Return(Close,5)`.
- **Fresh smoke**: seeds 8675901/8675902 completed the full one-step CPU chain at `artifacts/s1_gate_a_matrix_smoke/s1_gate_a_v1_merkle_final_cpu_v1/`; the complete index covered 141 files plus the self-excluded `matrix.json`, 16 immutable opening receipts verified, all arms had 20 canonical metric fields, no non-finite JSON occurred, and null promotions were zero.
- **Expected status**: `ineligible_residual_shapes` is one-step connectivity evidence only and not a Gate A result.
- **Custody finding**: the audit discovered that later pytest sessions automatically removed the historical `pytest-134` temporary tree. No manual or project deletion occurred. The original eight-file hash inventory remains historical evidence and the tree will not be reconstructed.
- **Correction**: added a separate custody addendum while preserving the original incident snapshot; scientific lock validation now binds both records.
- **Boundary**: this smoke used fresh seeds only. The project formal artifact directory remains absent; the invalid partial attempt still contains no test opening, and test-once entitlement remains unconsumed.
- **Next**: rebuild_lock_and_reverify

# S1 Gate A third independent-audit correction

## Iteration 10 (Phase 1/2: preserve E5 validation precheck and clear F841)

- **Score**: 0.89 (lint=1.00 format=0.00 test=1.00 self=0.98)
- **Red evidence**: fixed Ruff 0.12.12 reported exactly one F841 at `e5.py:518` because the validation design precheck result was assigned but never consumed.
- **Change**: retained `_structure_design(normalized, validation_matrix)` inside the existing `invalid_execution` transaction while removing only the unused assignment. This preserves validation shape/finite prechecking before least-squares accounting.
- **Lint**: `uvx --from ruff==0.12.12 ruff check .` passed with zero findings.
- **Format**: project-wide check remains nonzero because 37 pre-existing files do not match Ruff's formatter; no broad reformat was performed for this one-line audit fix.
- **Tests**: E5 target suite passed 6/6 under `-W error`.
- **Next**: final_regression_and_lock

## Iteration 11 (Phase 2/2: final lock, regression, CUDA, and fresh smoke)

- **Score**: 0.97 (lint=1.00 format=0.00 test=1.00 self=0.99; format is the documented pre-existing project-wide baseline)
- **Lint/tests**: fixed Ruff 0.12.12 passed with zero findings; compileall passed; full pytest passed 74/74 under `-W error` in 52.35 seconds.
- **CUDA**: NVIDIA A800-SXM4-80GB passed E4 float64 forward, regularized objective, backward gradients, analytical hardening, and promoted-HARD inference; every output/gradient was finite and the promoted source was `Return(Close,5)`.
- **Lock**: the one-line `e5.py` identity change was bound by implementation aggregate `d0e06454...a74b8` and lock SHA-256 `a57adb75...2c428`; timestamp and incident/addendum validation passed.
- **Fresh smoke**: fresh seeds 8675921/8675922 completed the full one-step CPU chain at `artifacts/s1_gate_a_matrix_smoke/s1_gate_a_v1_ruff_lockfinal_cpu_v1/`.
- **Integrity**: 141 recursively indexed files plus the self-excluded matrix manifest, 16 immutable receipts, all core-child hashes, seven reported arms with 20 canonical metrics each, and 68 JSON documents passed verification. Artifact aggregate: `83328198...cbd4a`; null promotions: zero.
- **Boundary**: `ineligible_residual_shapes` is connectivity evidence only. No formal project artifact exists, no frozen seed was resumed, the invalid historical partial attempt remains without test opening, and test-once entitlement remains unconsumed.
- **Next**: done

# S1 Gate A formal scientific execution and reflection

## Iteration 12 (sealed formal run and Plan C decision)

- **Formal run**: `artifacts/s1_gate_a_scientific/s1_gate_a_v1_scientific_formal_v1/`; seeds 1729/2718/31415; CUDA `cuda:0`; one exclusive test opening per seed.
- **Integrity**: 208/208 recursively indexed files; Merkle aggregate `63436d31...90a6b`; 24 immutable arm receipts, 3 matrix claims, and 18 selected-checkpoint hashes verified; no terminal failure; seal and implementation identities unchanged.
- **Gate result**: `scientific_fail`. Conditions 1/2/3/4/7 passed; conditions 5/6 failed because promotion status was `no_governed_candidate` and no HARD model existed.
- **Positive evidence**: E4 clean NRMSE median `0.0233467`; shape NRMSE median `0.0159897`; exact source/window recovered in all three seeds; null promotions zero.
- **Failure diagnosis**: exact-family fitting and nondup reference sanity passed. Joint analytical-plus-spline training produced a stable nonmonotone correction spline, not an identifiable missing primitive. The failure is architectural/scientific, not eligible implementation recovery.
- **Decision**: seal Gate A and activate staged Plan C. KAN becomes an optional semi-symbolic miner. Prioritize a real heterogeneous S2 factor library and pinned Quanta backtest; graph remains locked until factor-library value and miner redundancy are established.
- **Memory**: IVE recorded the residual-only additive-decomposition direction as a narrow fundamental failure with explicit boundary conditions.
- **Reports**: `reports/s1_gate_a_scientific_report.md`; `governance/decisions/2026-07-16_gate_a_plan_c.md`.
- **Next**: freeze S2 thresholds and build the real factor-library chain.

# S2a Plan C implementation (More Effort)

## Iteration 13 (4 phases: mining, publication, Quanta capture, verification)

- **Score**: 0.96 (lint=1.00 format=1.00 targeted_test=1.00 full_test=1.00 self=0.95; format applies to new S2 files only).
- **Implementation**: added deterministic 256-attempt typed mining, train/validation RankIC screening, fresh within-date permutation selection, label-free random control, greedy diversity/profile quotas, immutable three-library publication, exact Qlib `report_df`/signal interception, and no-replace Alpha158/custom evaluation commands.
- **Verification**: 85 tests collected and 85 passed under `/usr/bin/python -m pytest -W error -q --tb=short` in 54.32 seconds; targeted S2/publication regression passed 15/15; compileall passed; fixed Ruff 0.12.12 passed with zero findings; new S2 files pass format check.
- **Governance**: prospective S2 config, preregistration, lock, and proposal hashes remained unchanged. No real mining, Alpha158 replay, custom Quanta evaluation, S2 candidate outcome access, or result-bearing S2 artifact was produced.
- **Next**: independent implementation review and implementation-lock construction before any scientific/development execution.

## Iteration 14 (S2a pre-execution protocol hardening)

- **Score**: 0.89 (lint=1.00 format=0.00 test=1.00 self=0.98; project-wide format remains the documented pre-existing baseline and was not broadly rewritten).
- **Red evidence**: new behavioral/adversarial tests initially failed for aggregate-lookback provenance, undefined Spearman, manifest traversal/extra files, independent arm opening, replay-stop topology, dirty Quanta dependencies, stale/concurrent capture, missing transaction claims, wrong-arm evaluation placement, incomplete daily calendars, and all nine decision boundaries. Authority review corrected the initial 60-lag interpretation to the proposal-consistent 120-lag aggregate ceiling before execution.
- **Mining/publication**: the run root is now O_EXCL-claimed before identities, labels, or attempts; admission and mid-publication failures finalize immutable terminal evidence; selected/permuted admissions and deterministic random-cap-16 membership are recomputed; AST depth, nodes, required lookback, and required lag are separately ledgered; undefined diversity is rejected.
- **Evaluation**: result-bearing individual arm entry points are forbidden. One fixed-topology orchestrator creates a single immutable opening, runs the official Alpha158 replay first, stops on anchor mismatch, otherwise runs the three registered custom arms once, computes all nine frozen criteria and the exact four-year fairness rule, and publishes a report-hash-bound machine decision with `formal_promotion_allowed=false`.
- **Identity/integrity**: Quanta uses the real package import with exact `__file__`, clean tracked execution closure, package tree object, locked effective provider, full provider content identity, and serialized exact-backtest capture. Mining, library, and evaluation verifiers enforce safe flat filenames, root/category containment, exact file sets, current proposal/protocol/prereg/implementation/baseline/provider identities, arm roles, counts, provenance, and recomputability.
- **Tests**: full pytest passed 110/110 under `-W error` in 55.20 seconds; governed Gate A promotion regression independently passed 4/4 in 40.24 seconds; compileall passed; fixed Ruff 0.12.12 passed with zero findings.
- **Boundary**: no S2 labels, development-test results, mining outputs, Quanta runs, evaluation outputs, decisions, or reports were opened or produced. Frozen proposal, S2 config, preregistration, base lock, and S1 historical artifacts/locks were not edited.
- **Next**: construct the hash-bound S2 implementation lock, perform an independent read-only audit, then and only then execute the prospective mining/orchestrator chain.

## Iteration 15 (real-Qlib provider verification and implementation lock)

- **Red evidence**: a lock-after preflight in the actual Qlib 0.9.7 environment initialized the correct provider but raised `cannot verify Qlib's effective provider URI`; the fake config test had represented `provider_uri` as a scalar, while real Qlib stores a frequency map and resolves it through `C.dpm`.
- **Surgical fix**: provider verification now reads the structured effective path from `C.dpm.get_data_uri()` and retains strict fallback support for the legacy scalar/default-frequency mapping. No log parsing or path approximation was introduced.
- **Real-environment green**: the pinned runner imported from the exact Quanta repository; Qlib initialized at the locked provider; Quanta tree object, effective provider, and provider content-tree identity all verified without invoking training, prediction, or backtest.
- **Tests**: the new regression failed before the fix and passed afterward; full pytest passed 111/111 under `-W error` in 53.80 seconds; compileall passed; Ruff 0.12.12 passed with zero findings.
- **Implementation lock**: `prereg/s2_plan_c_vertical_v1_implementation.lock.json`; lock SHA-256 `eb8e070e...f461`; source tree `2251a501...d87b`; Qlib provider tree `1babf2a6...dca1` over 60,168 regular files and 632,291,118 bytes.
- **Boundary**: no mining, model training, prediction, backtest, S2 label outcome, or result artifact was produced.
- **Next**: repeat the independent read-only pre-execution audit against the rebuilt lock.

# S2a v2 pure-symbolic E3 core (More Effort)

## Iteration E3-1 (Phase 1/2: test-first contract freeze)

- **Score**: 0.46 (lint=1.00 format=0.00 test=0.00 self=0.65; expected red test-first state)
- **Lint**: passed (0 issues).
- **Format**: project-wide failed on 43 pre-existing/non-E3 files plus the new test; no broad formatting was applied.
- **Tests**: expected failure (0/11 targeted E3 tests) because `mirage_kan.mining.e3` did not yet exist; the broader suite reached its pre-existing tests before the targeted import failure.
- **Changes**: `tests/mining/test_e3.py`.
- **Feedback**: the red suite now fixes profile atom semantics, differentiable soft/hard-ST behavior, checkpoint-only hardening, hash tie-breaking, cancellation exclusion, objective/schedule/fidelity helpers, and invalid-input behavior before implementation.
- **Next**: implement the smallest E3 deep module that satisfies the frozen contracts.

## Iteration E3-2 (Phase 2/2: pure-symbolic E3 implementation)

- **Score**: 0.88 (lint=1.00 format=0.00 targeted_test=1.00 self=0.95).
- **Lint**: passed (0 issues in the new module and focused test).
- **Format**: failed only because the two new files had not yet been mechanically formatted.
- **Tests**: passed (11/11 focused E3 tests).
- **Changes**: `src/mirage_kan/mining/e3.py`, `tests/mining/test_e3.py`.
- **Feedback**: implemented four exact typed atom banks, two genuine differentiable categorical edges, fixed 300-step temperature/mode schedule, negative mean daily cross-sectional Pearson-IC objective, checkpoint-only hash-deterministic hardening with distinct atoms and rejection receipts, independent hard-AST replay, and shared fidelity metrics. No data, label, mining, Quanta, CLI, v1, config, preregistration, lock, or proposal path was touched.
- **Next**: format only the two new files, run full lint/regression, and perform the final API/authority self-audit.

## Iteration E3-3 (Phase 2/2: prospective protocol resync and full regression)

- **Score**: 0.99 (lint=1.00 format=1.00 test=1.00 self=0.97).
- **Protocol resync**: after the prospective v2 YAML was refined, new red assertions exposed both mismatches: bank order was generation order instead of canonical-hash order, and `lag_vs_price` divided by current price instead of delayed price. The implementation now follows the exact frozen family definitions, deduplicates before canonical-hash sorting, and assigns indices only after sorting.
- **Lint/format/compile**: fixed Ruff 0.12.12 passed project-wide with zero findings; both new files pass format checks; compileall passed.
- **Tests**: focused E3 passed 11/11; full regression passed 137/137 under `-W error` in 45.92 seconds.
- **Changes**: `src/mirage_kan/mining/e3.py`, `tests/mining/test_e3.py`, and this required iteration-log entry only.
- **Boundary**: no PIT labels, raw outcomes, mining, model fitting, Quanta, development opening, or result artifacts were accessed or executed; CLI, `mining/s2a.py`, all v1 artifacts, proposal, v2 config/preregistration/locks, and reference assets were not edited.
- **Next**: hand the verified E3 deep module to the parent pipeline for the separately governed runner/publication stage.

# S2a v4 implementation-lock GPU identity hardening (More Effort)

## Iteration LH-1 (Phase 1/2: test-first runtime identity contract)

- **Score**: 0.46 (lint=1.00 format=0.00 test=0.00 self=0.65; expected red test-first state).
- **Lint**: project-wide `ruff check src tests` passed with zero findings.
- **Format**: project-wide check retained the known 47 historical files outside this surgical change; no broad reformat was performed.
- **Tests**: the focused behavior test failed as intended because CUDA UUID, total memory, and both TF32 flags were absent from the live runtime identity.
- **Changes**: `tests/governance/test_implementation_lock.py` and this iteration entry only.
- **Next**: add the smallest fail-closed runtime capture, then run focused and broader regression checks.

## Iteration LH-2 (Phase 2/2: minimal capture and focused regression)

- **Score**: 0.60 (lint=1.00 format=0.00 test=1.00 self=0.92; capped before the one-file formatting correction).
- **Implementation**: each enumerated CUDA device now contributes its stable UUID and exact total-memory bytes; the determinism identity now binds CUDA matmul TF32 and cuDNN TF32 flags. Invalid empty UUID or non-positive memory fails before lock construction.
- **Lint/tests**: focused lint passed; implementation-lock tests passed 28/28; the broader governance suite passed 55/55.
- **Format**: the source file alone needed Ruff's mechanical line wrapping and was formatted without touching unrelated files.
- **Changes**: `src/mirage_kan/governance/implementation_lock.py`, `tests/governance/test_implementation_lock.py`, and this iteration entry.
- **Next**: verify the real two-GPU runtime identity and run the full regression suite plus final lint/format checks.

## Iteration LH-3 (Phase 2/2: real-runtime and full regression closure)

- **Score**: 0.98 (lint=1.00 format=1.00 test=1.00 self=0.96).
- **Real runtime**: Torch 2.9.1+cu129 exposed distinct stable UUIDs for both NVIDIA A800-SXM4-80GB devices; each reported exactly 85,167,243,264 bytes. The live identity captured CUDA matmul TF32=false and cuDNN TF32=true under the frozen v4 environment.
- **Tests**: implementation-lock tests passed 28/28, governance tests passed 55/55, and the complete suite passed 351/351 with only the existing pandas FutureWarning family.
- **Quality checks**: project-wide `ruff check src tests` passed; both changed Python files passed Ruff format check and compileall.
- **Boundary**: PCI bus identity was deliberately excluded because Torch does not provide it as a stable cross-version public contract. No configuration, preregistration, base lock, implementation lock, mining code, wheelhouse, or scientific artifact was changed.
- **Next**: done; parent may construct the v4 base and implementation locks from this hardened source identity.

# S2a v4 isolated runtime closure (More Effort)

## Iteration RC-1 (artifact closure and fail-closed publication)

- **Score**: 0.70; the initial sequential acquisition and missing-size assumption were replaced after measured failures.
- **Result**: selected and independently verified one hash-authorized artifact for each of 223 locked requirements (217 Linux-applicable), then published 4,884,582,030 bytes with exclusive-create and fsync semantics. Unsafe names, duplicates, symlinks, unexpected directories/files, hash drift, and overwrite attempts fail closed.
- **Boundary**: no scientific/config/preregistration/governance/test file was edited.

## Iteration RC-2 (production compatibility and import/launcher closure)

- **Score**: 0.93; runtime-tool lint, format, and all focused compatibility gates passed.
- **Result**: exact production AST/checkpoint publication parity passed on two A800 GPUs; full production E3 and MLP-control training paths completed. The launcher now directly enters v4 mining/development with exact environment, project-only import path, bytecode suppression, deterministic Torch state, and symlink-safe pre-Qlib tracking CWD. Complete production entrypoint imports leave `qlib.config.C.registered=false` and do not access labels.
- **Boundary**: synthetic compatibility data only; no real provider initialization, mining, or development opening.

## Iteration RC-3 (offline rebuild, regression, and categorized seal)

- **Score**: 0.98 (lint=1.00, format=1.00, focused=1.00, regression=1.00, self=0.96).
- **Offline proof**: a fresh Python 3.12.3 venv installed 217 packages using only `--offline --no-index --require-hashes`; dependency checking and the exact Torch 2.9.1+cu129/CUDA 12.9/cuDNN 9.10.2.21/two-GPU assertions passed.
- **Regression**: after final live-runtime lock hardening, the exact-v4 full suite passed 368 tests with 933 warnings in 129.95 seconds; Ruff checks passed; seven active files passed format check.
- **Seal**: manifest-last JSON/TSV evidence binds 20 categorized runtime files, 223 artifacts, and 217 installed RECORD identities. Acquisition duplicates and runtime-tool bytecode caches were removed before the final manifest generation.
- **Next**: bind final runtime launcher/helper identities in governance locks before formal execution.
