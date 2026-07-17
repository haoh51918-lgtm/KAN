# Evolution Report: Cycle 1 — IVE

> Historical cycle record, not a current route or instruction. Use the Living Manual and `plans/todos.md` for current work.

**Date**: 2026-07-16  
**Trigger**: S1 Gate A v1 completed as `scientific_fail` because executable primitive promotion and the interpretability Pareto gate failed.  
**Source Artifacts**: `KAN_Alpha_PR.md`; formal Gate report; promotion manifest; three frozen residual arrays; independent result audit; independent failure diagnosis.

## Changes Made

### Added

- Added `Residual-only primitive promotion under jointly trained additive decomposition` to Ideation Memory's unsuccessful directions.

### Updated

- None.

### Removed/Archived

- None.

## Reasoning

The paper-style IVE result is a failed validation of the complete proposal path: the method was executable and numerically better than the baselines, but it failed the preregistered executable-output requirement. The extended diagnostic separates the strong E4 predictor from the failed residual-only promotion hypothesis.

The failure is stored narrowly. It does not say KAN is universally useless. It says a jointly optimized additive analytical-plus-spline edge does not make the spline component mechanism-identifiable without an additional constraint. This abstraction is reusable in any architecture that later tries to interpret one term of a non-identifiable additive decomposition.

## Impact on Future Cycles

- **For research-ideation**: Prune proposals that assume a stable residual automatically equals a missing mechanism. Allow distinct branches that enforce decomposition identifiability or promote a full edge.
- **For experiment-pipeline**: Add decomposition sanity plots and exact-family self-fit checks before expensive promotion runs. Treat soft numerical success and hard executable success as separate gates.
- **Confidence level**: High for the frozen residual-only design: three seeds, exact reference sanity, independent artifact audit, and consistent failure geometry.

## Raw Evidence Summary

E4 passed source recovery and achieved shape NRMSE near `0.016`, but no HARD model was produced. Exact-family self-fit NRMSE was `0`, while the three formal residual-family fits were approximately `0.59–0.73`. The residual was a nonmonotone correction and the analytical gates remained mixed. Gate 5 and 6 therefore failed exactly as preregistered.
