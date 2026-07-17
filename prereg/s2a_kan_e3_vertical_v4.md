# S2a MIRAGE-KAN adaptive complete-chain preregistration v4

## Status and authority

This is a prospective developmental-screen successor under the sole authority
of `KAN_Alpha_PR.md`, especially Section 25. It is informed only by the v3
pre-development fact that seven strict factors passed against a count floor of
eight. No v3 factor identity, candidate score, development outcome, portfolio
metric, or Quanta result was published or observed.

The authorization and custody record is
`governance/incidents/2026-07-17_s2a_v4_adaptive_successor.md`. WIKI and prior
result directories remain non-authoritative references. v4 cannot establish a
paper claim, formally release an interpretable factor library, or reuse the
2022--2025 period as final confirmation evidence.

## Research question

Can a non-graph population of genuine pure-symbolic E3 KAN miners produce a
strict, fully traceable factor library whose cost-aware Quanta portfolio is
non-inferior to Alpha158 and a matched-capacity black-box control, while not
being jointly Pareto-dominated by a matched-budget typed GP/SR control?

The primary horizontal outcome is the backtest quality of the generated factor
library. Search time, training iterations, and library count are secondary
diagnostics. Interpretability cannot excuse statistically material
underperformance.

## Adaptive change from v3

The complete executable specification is
`configs/experiments/s2a_kan_e3_vertical_v4.yaml`. After removing protocol ID
and artifact paths, it must equal the v3 parsed YAML object except for exactly
two values:

- `admission.minimum_library_size`: 8 → 6;
- `s2a_decision.integrity.production_library_size_minimum`: 8 → 6.

Six is a developmental floor with two factors per required profile on average;
it is intentionally not fitted to the observed v3 count seven. The cap remains
16 and at least three profiles remain required. Every per-factor efficacy,
coverage, train/validation sign, fidelity, gate-margin, strict-formula,
independent-replay, and diversity rule remains unchanged.

All four profiles, 64 miners per profile, 256 production attempts, 256 complete
within-date permutation attempts, 256 typed GP/SR attempts, 300 Adam updates,
seeds, bootstrap rules, atom definitions, control semantics, mechanism tests,
Quanta costs, portfolio settings, 10,000-replicate paired bootstrap, guardrails,
and performance thresholds remain unchanged.

## Production and controls

The production miner remains the real differentiable E3 categorical-edge KAN
over internally evaluated typed temporal atoms. It hardens exactly one
`Sub(atom_positive, atom_negative)` formula per attempt and the common DSL
executor recomputes that AST independently. No handwritten formula, complete GP
formula, predecessor member, hidden retry, top-k alternate, spline residual, or
externally precomputed alpha input may enter production.

The fixed controls remain:

1. official Alpha158 replay;
2. the selected KAN E3 library;
3. same-DSL, same-budget typed GP/SR;
4. one two-unit SiLU MLP paired to each selected KAN factor with the same atom
   bank and bootstrap weights; and
5. a complete KAN rerun using within-date permuted labels.

Control factors are never promotion eligible.

## Data and access discipline

- Train: 2016-01-01 through 2020-12-31.
- Validation: 2021-01-01 through 2021-12-31.
- Development portfolio test: 2022-01-01 through 2025-12-26.
- Label: exact Quanta `fwd`.

The final two trading dates of train and validation are physically excluded
from label reads and objectives. Rolling features receive exactly 60 preceding
raw-only trading dates; warm-up labels are null. The mining entitlement must be
consumed before first label access. Mining outputs contain no development raw
values or predictions.

All production and control mining artifacts must be immutable before the one
development opening. After that opening, frozen ASTs and MLP final parameters
are replayed point-in-time on the full raw panel. Any authority, predecessor
custody, implementation, data, Quanta, provider, or runtime drift stops the next
scientific receipt and forbids publication.

## Complete runtime closure

v4 uses only the isolated environment under `runtime/s2a_v4_eval/`. It may not
concatenate, inherit, or dynamically add another environment's `site-packages`.
Before label access, the implementation lock must bind:

- Python 3.12.3 and the executable hash;
- hash-locked requirements, resolver lock, wheelhouse manifest, and environment
  manifest;
- every installed distribution's canonical name, exact version, and
  `.dist-info/RECORD` SHA-256;
- Torch 2.9.1+cu129, CUDA, cuDNN, deterministic flags, and GPU identity;
- pyqlib 0.9.7, LightGBM 4.6.0, and MLflow 3.14.0;
- Quanta commit/config/runner and the complete QLib provider tree; and
- `CUBLAS_WORKSPACE_CONFIG`, `QLIB_DATA_DIR`, and
  `MLFLOW_ALLOW_FILE_STORE`.

An offline install from the local wheelhouse, dependency check, complete import
smoke, and train/validation-only AST/MLP replay gate must pass before the
implementation lock is issued. A failed compatibility gate cannot be repaired
by relaxing numerical tolerances.

## Development decision

All five arms use the pinned real Quanta path, LightGBM 500-round cap with early
stopping 50, TopkDropout 50/5, open-price execution, and frozen costs. The
primary statistic remains the one-sided 95% lower confidence bound of paired
development Information-Ratio difference. Alpha158 replay tolerances, KAN-minus-
Alpha158 and KAN-minus-MLP non-inferiority gates, RankIC, drawdown, turnover,
cost, calendar-year, permutation, GP/SR Pareto, mechanism, and effective-rank
rules remain exactly as specified in v3 YAML.

If every autonomous gate passes but genuine human blind review is pending, the
strongest outcome is `advance_s2_formal_pending_human_blind_review`. Formal
promotion remains forbidden in v4.

## Required artifacts

- proposal/config/preregistration/base/implementation/runtime lock chain;
- append-only authority, access, KAN, GP/SR, permutation, MLP, rejection, and
  custody evidence;
- immutable production and control libraries/panels with exact provenance;
- one mechanism card per selected KAN factor and anonymized blind package;
- one current Alpha158 replay and four real custom Quanta evaluations;
- daily returns, turnover, costs, prediction coverage, bootstrap draws, yearly
  summaries, effective rank, integrity gates, and a single decision artifact;
- terminal aggregate rejection diagnostics if mining fails before publication.

