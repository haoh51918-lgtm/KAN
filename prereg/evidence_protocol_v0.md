# Evidence Protocol v0

> This document freezes comparison semantics before the first project-generated factor library is evaluated. Numerical method thresholds will be versioned before their owning experiment; they must not be invented after seeing the result.

## Frozen identities

- Proposal: `KAN_Alpha_PR.md`.
- QuantaAlpha revision: `b7ceb27b1001261d7a95b209a963664ae1f8ab23`.
- Backtest config SHA-256: `4e095512025a44dcca279e3d3c4d02fc83367caf044032b6c9f6eeb94405a832`.
- Backtest runner SHA-256: `a18ec5bfbe57b452dbacb3cdd15249f99c2b53e7c0761c178e9fbb89db7d34d8`.
- PIT cache SHA-256: `cbcf1c0e06f0a966f503d9c2fc1688fbe78faee5a9a46b99f32abf9498229f69`.
- Market: CSI300.
- Label: `Ref($close, -2) / Ref($close, -1) - 1`.
- Train / validation / development evaluation: 2016–2020 / 2021 / 2022–2025.
- Model: Quanta LightGBM, at most 500 rounds, early stopping 50.
- Portfolio: TopkDropout top 50, drop 5, open-price transaction-cost model.

## Primary comparison

The primary estimand is the paired change in net portfolio Information Ratio produced by replacing or augmenting the selected baseline factor library while keeping the evaluator fixed. The exact primary baseline arm and common factor cap are frozen in the S2 preregistration after S0 establishes adapter parity and before any S2 result is read.

## Mandatory invariants

1. Universe membership, raw-field observation, expression support, finite output, and tradability are separate concepts. Missing observations are never filled into membership-valid evidence.
2. Every operator declares output type, lookback, numerical domain, and support propagation. Future access is invalid at compile time.
3. Candidate selection uses training pseudo-future folds only. Validation is shared model selection/early stopping; 2022–2025 access is logged as development evidence.
4. Every generated candidate receives a ledger entry: accepted, invalid syntax/type/domain, non-finite, duplicate, low fidelity, screened out, fully evaluated, or failed.
5. All comparison arms share factor cap, joint support, processors, LightGBM, backtest, costs, seeds, and matched candidate/full-evaluation budgets where the arm is a search method.
6. Label permutation reruns the complete search-to-portfolio procedure. Permuting only the final prediction is not a valid negative control.
7. Historical metrics are linked with source/config/data identity. They are not silently reinterpreted under a new protocol.

## Promotion and backtracking

- S0 failures are implementation/data failures, not evidence against the method.
- Gate A is evaluated only after the out-of-dictionary mechanism test. Failure prevents graph expansion but does not prevent a KAN-decentered control study.
- S2 must show real factor-library value before graph work is unlocked.
- A graph claim requires superiority to the same-information Flat-Controller and relevant non-graph adaptive controls; superiority to random alone is insufficient.
- Strict mechanism claims require frozen soft/hard and intervention fidelity. Failing candidates are explicitly classified as semi-symbolic or neural.
- Any material null-control signal, unfair support, or data leakage invalidates the owning result and returns the pipeline to S0.

## Development evidence ceiling

The 2022–2025 interval has influenced prior baseline work and is not an untouched lockbox. It supports development comparison and honest model iteration. Final confirmation requires a separately frozen future period or market that does not influence method, thresholds, or narrative.

