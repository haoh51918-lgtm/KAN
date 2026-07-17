# S2a MIRAGE-KAN corrective preregistration v3

## Status

This is a prospective developmental-screen successor under the sole authority
of `KAN_Alpha_PR.md`, especially Section 25. It cannot establish a paper claim,
formally release an interpretable factor library, or use the 2022--2025 period
as final confirmation evidence.

S2a v2 terminated in a pre-label infrastructure failure when typed YAML dates
could not be serialized into the single-use mining entitlement. No label,
candidate, factor library, control result, development outcome, or Quanta result
was observed. v3 therefore preserves the v2 scientific design exactly and uses
new ownership paths throughout. The frozen custody and authorization record is
`governance/incidents/2026-07-17_s2a_v3_corrective_successor.md`.

## Frozen scientific protocol

The complete executable scientific specification is
`configs/experiments/s2a_kan_e3_vertical_v3.yaml`. Relative to the v2 frozen
configuration, the only allowed differences are:

1. `protocol_id` changes from v2 to v3;
2. every writable artifact, opening, recovery, evaluation, decision, report,
   and implementation-lock path changes to a disjoint v3 path;
3. opening-receipt dates are canonically represented as ISO strings;
4. the complete JSON payload is encoded before an exclusive destination is
   created; and
5. Quanta experiment and recorder names are derived from the active protocol ID.

All other fields must compare equal as parsed YAML objects. In particular, data
splits, the exact two-trading-day horizon purge, 60-date raw-only warm-up, four
KAN profiles, 256 production attempts, 256 typed GP/SR attempts, 256 permutation
attempts, 300 Adam steps, seeds, admission thresholds, library size 8--16,
mechanism evidence, five Quanta arms, 10,000-replicate paired bootstrap, costs,
guardrails, and S2a decision thresholds are unchanged.

The production miner remains a genuine categorical-edge E3 KAN over internally
evaluated typed temporal atoms. It hardens exactly one strict
`Sub(atom_positive, atom_negative)` formula per attempt and independently
replays the AST. GP/SR is a matched-budget non-production method control; the
two-unit SiLU MLP is a matched-capacity non-production falsification control;
the within-date permutation run is a complete negative-control rerun. Neither a
handwritten formula nor a predecessor candidate may enter the v3 library.

## Data and opening discipline

- Train: 2016-01-01 through 2020-12-31.
- Validation: 2021-01-01 through 2021-12-31.
- Development portfolio test: 2022-01-01 through 2025-12-26.
- Label: exact Quanta `fwd`, with the last two trading dates of train and
  validation physically excluded from label reads and objectives.
- Feature warm-up: exactly the prior 60 raw-only trading dates, with labels null.

The mining entitlement must be consumed before first label access. All mining
and control artifacts must be immutable before the single development opening.
After that opening, frozen ASTs and MLP parameters are replayed point-in-time on
the full raw panel and evaluated through the pinned Quanta implementation.

## Primary success criterion

The main horizontal comparison is the backtest quality of the generated factor
library, not candidate count, training time, or number of iterations. The
primary statistic remains the one-sided 95% lower confidence bound of paired
development Information-Ratio difference, with Alpha158 replay, matched MLP,
typed GP/SR, permutation, integrity, turnover, cost, drawdown, RankIC, calendar-
year, diversity, and mechanism gates all applied exactly as frozen in YAML.

Even if every autonomous gate passes, the strongest v3 outcome is
`advance_s2_formal_pending_human_blind_review` until at least two genuine human
quantitative reviewers complete the frozen blinded mechanism review. Formal
promotion remains forbidden in v3.

## Pre-label execution gates

Before any v3 label is read:

1. the v2 custody evidence must remain untouched;
2. the v2/v3 parsed scientific settings equality test must pass;
3. the v2/v3 writable-path disjointness test must pass;
4. typed-date and encode-before-create regression tests must pass;
5. the full deterministic test suite and Ruff must pass; and
6. a new v3 implementation lock must bind the complete source/runtime/data/
   Quanta/provider closure.

Any authority drift or custody violation makes the run inconclusive and forbids
continued publication.

