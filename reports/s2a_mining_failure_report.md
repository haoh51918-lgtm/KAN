# MIRAGE-KAN S2a mining failure report

## Outcome

`FAILED (NoExecutableWithinBudget)` before development-test access. The fixed four-arm topology could not be constructed because the label-permutation arm produced no candidate passing the common RankIC admission floor.

## Reproducible setting

- Protocol: `s2_plan_c_vertical_v1`
- Candidate budget: 256, exactly 64 per miner profile
- Train: 2016–2020
- Validation: 2021
- Development test: unopened
- Implementation lock: `297624791e2d3766557540c8545bcc4d0cc110c7bee5069b17e716eb7df9b40c`
- Mining root: `artifacts/s2a_plan_c_mining_v1/`

## Failure-case inventory

| Arm | Unique eligible before diversity | Selected | Profile coverage | Status |
|---|---:|---:|---:|---|
| Observed-label selected | 178 | 16 | 4/4 | Working |
| Label-free random | 16 control members | 16 | 4/4 | Working |
| Within-date label permutation | 0 | 0 | 0/4 | Admission failure |

For the permutation scorer, 217 unique candidates were `ineligible_rank_ic` and 39 attempts duplicated a canonical identity. The failure occurred before diversity or profile balancing, so those constraints are not causal.

## Five-step diagnosis

1. **Failure cases**: all unique permutation candidates failed the absolute predictive-signal floor.
2. **Working versions**: observed selection and label-free random selection both reached 16 factors with all four profiles.
3. **Isolated difference**: applying destroyed labels while retaining the method arm's efficacy floor eliminated the entire null candidate pool.
4. **Verified hypothesis**: the null labels behaved as intended; the shared admission semantics, not scorer execution, made the control non-executable.
5. **Proposed fix**: a prospectively frozen size-matched permutation control that retains structural validity, coverage, fresh scoring, deterministic ranking, diversity, and profile balance, but does not require destroyed labels to demonstrate real predictive efficacy.

## Integrity

The mining run and all three claimed factor-library directories are terminal artifacts with no `.INCOMPLETE` marker. The mining manifest records all 256 attempts and hashes the observed, permutation, random, and data-access ledgers. No 2022–2025 evaluation, decision, or report was created.

## Claim boundary

The observed training/validation screen is encouraging but is not portfolio evidence. No statement about outperforming Alpha158 or producing a higher-quality alpha library is permitted from this run.
