# S1 Gate A Capacity Harness — Stage 1 Implementation Trajectory

These are tiny implementation smokes only. They do not consume or report the
frozen three-seed Gate A scientific matrix.

## Attempt 1 — module invocation preflight

**Hypothesis**: The source-layout smoke module can be invoked directly.

**Code Changes**: None.

**Configuration**: E1, CPU, 3 steps, tiny non-frozen panel.

**Result**: Failed before generation because the uninstalled source layout was
not on Python's import path. Raw output is preserved in
`logs/s1_gate_a_e1_tiny_cpu/console.log`.

**Analysis**: Invocation, not model or data code, caused the failure. The exact
run command must use `PYTHONPATH=src` unless the package is installed.

## Attempt 2 — E1 CPU integration smoke

**Hypothesis**: The corrected source-path command will exercise the complete
generated-data → E1 → validation checkpoint → test-once artifact path.

**Code Changes**: None; command-only recovery.

**Configuration**: E1, CPU float64, 3 steps, seed 41, 4 assets, 32/8/8 dates.

**Result**: Completed with a 115-parameter checkpoint, 3 optimizer steps, 384
sample presentations, and all categorized artifacts present. Scientific metric
values are intentionally not interpreted here.

**Analysis**: Hypothesis confirmed. The common harness works on CPU.

## Attempt 3 — seeded C6 CUDA integration smoke

**Hypothesis**: A prospectively seeded C6 initialization plus the same harness
will reproduce the integration path on available CUDA.

**Code Changes**: C6 initialization now uses a local seeded Torch RNG without
mutating the caller's global RNG; test-once reloads the selected checkpoint.

**Configuration**: C6, CUDA float64, 3 steps, seed 41, 4 assets, 32/8/8 dates.

**Result**: Completed with the declared 361-parameter checkpoint, 3 optimizer
steps, 384 sample presentations, and all categorized artifacts present.
Scientific metric values are intentionally not interpreted here.

**Analysis**: Hypothesis confirmed. **[Reusable]** Source-layout experiment
commands should state `PYTHONPATH=src`, and model initialization identity must be
recorded separately from the minibatch RNG seed.
