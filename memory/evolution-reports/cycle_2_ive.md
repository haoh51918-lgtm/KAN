# Cycle 2 IVE — S2a permutation-control admission failure

> Historical cycle record. Its retry recommendation was superseded by later S2a v2-v8 evidence; it must not trigger v9.

## Classification

- **Paper outcome**: `FAILED (NoExecutableWithinBudget)`
- **Failure class**: implementation/protocol failure, retryable
- **Research direction**: retained as feasible with a control-design correction

## Evidence

The frozen S2a budget completed all 256 candidate executions. The observed-label scorer found 178 eligible candidates and the greedy selector produced 16 factors spanning trend, mean reversion, price-volume, and typed-composition profiles. The label-free random control also produced 16 factors across all four profiles.

The within-date label-permutation scorer found zero candidates passing the common absolute train/validation RankIC floor. All 217 unique candidates were `ineligible_rank_ic`; 39 other attempts duplicated a canonical identity. The transaction therefore emitted `S2a permutation admission is below the frozen minimum`, terminalized the mining run and all three library claims, and never created a development opening.

## Confirmed root cause

The protocol used one efficacy admission rule for both the proposed method arm and the destroyed-label null arm. A successful permutation destroys predictive signal, so requiring at least eight null candidates to exceed the same absolute signal floor makes control construction structurally unreliable. Profile quota and diversity were not the cause because zero candidates reached the selection stage.

## Memory update

At the time, the cycle added `Size-matched null controls for thresholded factor-library selection` with status `retry with fixes`. That status is now closed/superseded in `memory/ideation-memory.md`.

## Historical countermeasures (consumed by later cycles)

1. Keep every observed-arm threshold unchanged.
2. For the permutation null only, separate structural/coverage eligibility from predictive-efficacy admission.
3. Preserve fresh permutation scoring, deterministic ranking, profile quota, diversity, and exact size matching.
4. Reuse the exact sealed 256-attempt topology; do not authorize another candidate search.
5. Prove observed and random memberships are unchanged under the successor protocol.
6. Freeze the successor implementation before any 2022–2025 result access.

## Expected impact

Future factor-library experiments will distinguish “method admission” from “negative-control construction.” A null that correctly contains no threshold-passing signal will no longer prevent the full comparison from being executable, while the proposed method's admission standard remains intact.
