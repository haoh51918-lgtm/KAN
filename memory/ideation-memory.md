# Ideation Memory

Last Updated: 2026-07-17
Total Cycles: 2

> Historical scientific memory, not a current action queue. Current work is in `plans/todos.md`; later evidence supersedes older retry guidance.

## Closed directions

### Size-matched null controls for thresholded factor-library selection

- **Summary**: Factor-library null controls must repeat candidate generation, scoring, ranking, diversity, and profile balancing, but must not require the null labels to pass the method arm's absolute predictive-signal admission floor.
- **Why Promising**: The real observed-label screen produced 178 eligible candidates and a diverse 16-factor library, while the correctly destroyed-label screen produced zero eligible candidates. A size-matched null can preserve the scientific comparison without weakening the observed method arm.
- **Requirements**: Keep the selected arm's frozen coverage, train/validation RankIC, sign, diversity, and profile rules unchanged. For the permutation control only, admit structurally valid and sufficiently covered candidates, rank them by the same score/tie-breakers, and select an equal-size diverse library while explicitly recording that they failed the efficacy floor.
- **Historical Validation Plan**: This was the proposed recovery after S2a v1. Later S2a v2-v8 superseded and exhausted this route; it must not create another opening.
- **Evidence**: S2a cycle 2 used exactly 256 attempts. Observed labels yielded 178 eligible candidates and selected 16 across all four miner profiles. Within-date permuted labels yielded 217 unique candidates, all `ineligible_rank_ic`, so the four-arm transaction terminated before test access. Random label-free selection produced 16 candidates across four profiles.
- **Status**: closed and superseded; no S2a v9 retry
- **Do-Not-Retry Guidance**: Preserve this as a lesson about control-specific eligibility. Do not reconstruct another successor, lower thresholds, reuse terminal libraries, or use quarantined v8 metrics. Revisit the concept only in a genuinely new experiment justified by the current science queue.
- **Countermeasures**: Separate method efficacy admission from null size matching; preregister control-specific eligibility; require exact library-size parity; retain full permutation label/scoring hashes; keep profile and diversity constraints; fail closed if the successor changes observed/random membership.
- **Retry Count**: 1
- **Retrieval Tags**: factor-library, permutation-null, negative-control, size-matching, RankIC-threshold, symbolic-mining, S2a
- **Date Added**: 2026-07-17
- **Last Updated**: 2026-07-17

## Unsuccessful Directions

### Residual-only primitive promotion under jointly trained additive decomposition

- **Summary**: Jointly training a soft analytical mixture and a free spline does not identify the spline as the missing monotone primitive, even when the total model recovers a simple out-of-dictionary mechanism accurately.
- **Failure Classification**: Fundamental for the frozen residual-only decomposition; this does not classify all KAN use as fundamentally invalid.
- **Evidence**: Gate A v1 completed with E4 shape NRMSE `0.015990 / 0.014267 / 0.016125`, but all frozen promotion families had median residual-fit NRMSE about `0.669` and no governed HARD model was produced. The exact exponential family self-fit recovered `[0.6, 2.5, 1.8, 0.25]` with NRMSE `0`, excluding fitter failure.
- **Diagnostic Answers**: Q1 partial total-model success, but no residual-only success; Q2 the residual-only isolation hypothesis failed on the simplest controlled mechanism; Q3 related whole-edge symbolification does not establish additive residual identifiability; Q4 the same nonmonotone correction appeared across all three seeds; Q5 no implementation bug was found. Four of the five signals support a conceptual decomposition failure.
- **Root Cause**: The analytical mixture and spline are additively non-identifiable. Optimization assigns most of the target shape to a mixture of `Clip`, `NegativeHinge`, and `Abs`, leaving the spline as a stable nonmonotone error-correction curve rather than the hidden primitive.
- **Boundary Conditions**: The direction may become viable if decomposition is made identifiable through hard-first or sequential training, orthogonality constraints, or promotion and replacement of the complete selected edge rather than the residual alone. Those are new methods, not threshold changes.
- **Countermeasures**: Require exact-family self-fit tests; inspect analytical/residual/total decomposition before promotion; never infer mechanism identity from low residual energy; require a frozen hardening-fidelity gate; preregister any full-edge or sequential variant with new test seeds.
- **Do-Not-Repeat Notes**: Do not lower nondup or approximation thresholds after failure. Do not promote a stable residual solely because cross-seed correlation is high. Do not reuse opened v1 test splits.
- **Retrieval Tags**: KAN, symbolic-residual, additive-identifiability, primitive-promotion, hardening, semi-symbolic, Gate-A
- **Date Added**: 2026-07-16

## Archive

*No archived entries.*
