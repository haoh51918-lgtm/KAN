# S2a MIRAGE-KAN calendar-corrective complete-chain preregistration v5

## Status and authority

`KAN_Alpha_PR.md` is the sole proposal authority. WIKI and prior result
directories are non-authoritative references. The governing incident is
`governance/incidents/2026-07-17_s2a_v5_calendar_corrective_successor.md`.

v5 follows a v4 infrastructure failure after the 2022--2025 development period
was opened. It is therefore a corrective adaptive repeated-development screen,
not a fresh prospective one-shot, independent confirmation, or paper claim.
Formal promotion and graph unlock remain forbidden.

## Research question and primary outcome

Can the genuine pure-symbolic E3 KAN miner produce a strict, traceable factor
library whose fixed, cost-aware Quanta portfolio is non-inferior to Alpha158 and
a matched-capacity black-box control, while not being jointly Pareto-dominated
by matched-budget typed GP/SR?

The primary horizontal outcome is factor-library backtest quality. Library
count, training updates, runtime, and compute utilization are diagnostics only.

## Frozen scientific specification

The executable specification is
`configs/experiments/s2a_kan_e3_vertical_v5.yaml`. After removing protocol ID,
writable artifact paths, and evidence-class metadata, its parsed YAML object
must equal v4 exactly. In particular:

- train is 2016-01-01 through 2020-12-31;
- validation is 2021-01-01 through 2021-12-31;
- repeated development is 2022-01-01 through 2025-12-26;
- the final two trading dates at train and validation boundaries remain purged;
- feature warm-up remains 60 raw-only trading dates;
- production, permutation, and typed GP/SR each receive 256 attempts;
- each KAN miner receives 300 Adam updates in float64;
- the production library must contain 6--16 factors over at least 3 profiles;
- every coverage, RankIC, sign, fidelity, gate-margin, strict-AST, replay, and
  diversity threshold remains unchanged;
- five arms remain Alpha158, selected KAN, typed GP/SR, matched MLP, and full
  within-date label-permutation KAN, in that order;
- Quanta model, portfolio, costs, metric definitions, 10,000-replicate paired
  bootstrap, guardrails, and decision thresholds remain unchanged.

v5 reruns the full mining topology. No v4 mining factor, development prediction,
label, model state, metric, portfolio output, or early-stopping observation is
reused as an active scientific input.

## Exact infrastructure correction

The pinned Quanta source, commit, configuration, and runner remain unchanged.
Its custom precomputed-data handler requires tuple ranges, while YAML supplies
lists. For custom-factor arms only, the adapter must:

1. require exactly the frozen `train`, `valid`, and `test` two-date sequences;
2. validate nonempty string dates and forward order;
3. preserve every date value and its order;
4. temporarily expose equal-value tuples during dataset construction;
5. restore the original runner configuration in `finally`; and
6. record the unchanged values in execution identity.

The official Alpha158 call must not use this correction. Signal cropping,
coverage reindexing, calendar-superset acceptance, or any test-only workaround
is forbidden. The constructed dataset must prepare mutually disjoint train,
valid, and test partitions, and the raw prediction signal must itself cover
exactly the frozen development calendar.

## Runtime and access discipline

The byte-identical v4 isolated Python environment and offline wheelhouse may be
reused as a dependency closure. The v5 launcher, tracking path, synthetic segment
smoke, regression evidence, base lock, and implementation lock are new and
disjoint. The implementation lock must rebind all source files, installed
distribution RECORDs, Python, Torch/CUDA/cuDNN, GPUs, deterministic flags,
environment variables, Quanta pins, PIT cache, baseline, and the complete Qlib
provider tree.

Mining entitlement is consumed before first v5 label access. All seven mining
topology targets must publish immutably before the single v5 development opening.
All five evaluation arms first stage privately; any arm failure cleans all
staging and terminalizes the complete topology. No partial arm may become a
scientific result.

The v4 MLflow tree is quarantined and used only as hash-bound failure custody.
v5 uses `evaluations/runtime/s2a_v5_tracking` and must not read or append v4
tracking runs.

## Decision boundary

The primary statistic remains the one-sided 95% lower confidence bound of the
paired development Information-Ratio difference. Alpha158 replay tolerances,
KAN-minus-Alpha158 and KAN-minus-MLP non-inferiority, RankIC, drawdown, turnover,
cost, yearly, permutation, GP/SR Pareto, mechanism, effective-rank, and integrity
gates remain frozen.

If autonomous gates pass while genuine human blind review is pending, the
strongest allowable status is an adaptive screen recommendation pending review.
It cannot be called prospective confirmation or formal promotion. A subsequent
claim requires an unseen period or market.

## Required outputs

- complete v5 authority, custody, runtime, base-lock, and implementation-lock chain;
- immutable KAN, GP/SR, permutation, matched-MLP, mechanism, and blind-review mining artifacts;
- five current Quanta evaluations with exact daily return, turnover, cost, and prediction coverage;
- bootstrap draws, yearly summaries, effective-rank and all integrity criteria;
- one atomically published decision and a human-readable Chinese report;
- terminal evidence, not partial publication, for any failure.
