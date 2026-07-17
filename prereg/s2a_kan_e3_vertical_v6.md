# S2a MIRAGE-KAN lineage-corrective complete-chain preregistration v6

## Status and authority

`KAN_Alpha_PR.md` is the sole proposal authority. WIKI and prior result
directories are non-authoritative references. The governing incident is
`governance/incidents/2026-07-17_s2a_v6_lineage_corrective_successor.md`.

v6 follows a v5 infrastructure failure after the 2022--2025 development period
was opened. It is a corrective adaptive repeated-development screen, not a
fresh prospective one-shot, independent confirmation, or paper claim. Formal
promotion and graph unlock remain forbidden.

## Research question and primary outcome

Can the genuine pure-symbolic E3 KAN miner produce a strict, traceable factor
library whose fixed, cost-aware Quanta portfolio is non-inferior to Alpha158 and
a matched-capacity black-box control, while not being jointly Pareto-dominated
by matched-budget typed GP/SR?

The primary horizontal outcome is factor-library backtest quality. Library
count, training updates, runtime, and compute utilization are diagnostics only.

## Frozen scientific specification

The executable specification is
`configs/experiments/s2a_kan_e3_vertical_v6.yaml`. After removing protocol ID
and writable artifact paths, its parsed YAML object must equal v5 exactly.
Train, validation, repeated-development dates, boundary purge, raw-only warm-up,
search space, profiles, 256-attempt budgets, seeds, 300-update optimization,
admission, diversity, controls, Quanta model, portfolio, costs, metrics,
bootstrap, guardrails, and decision thresholds are unchanged.

The production library must contain 6--16 strict factors spanning at least
three profiles. The five development arms remain Alpha158, selected KAN, typed
GP/SR, matched MLP, and full within-date label-permutation KAN in that order.

v6 reruns the full mining topology. No v5 mining factor, development prediction,
label, model state, metric, portfolio output, or early-stopping observation is
reused as an active scientific input.

## Exact infrastructure corrections

The pinned Quanta source, commit, configuration, and runner remain unchanged.
The v5 calendar correction remains mandatory: custom-factor train, valid, and
test ranges are validated as frozen two-date sequences, temporarily exposed as
equal-value tuples only during dataset construction, and restored in `finally`.
No prediction cropping, reindexing, or calendar relaxation is allowed.

The v6 lineage correction changes only identity validation. For matched MLP
controls, the loader must compare the complete factor-ID-to-global-attempt
mapping against selected KAN lineage. It must not require selection order to
equal canonical dictionary order and must not reorder prediction columns,
receipts, trajectories, seeds, or bootstrap bindings. A crossed assignment
with unchanged ID and index sets must fail closed. The top-bundle publisher
must enforce the same one-to-one mapping contract.

## Runtime and access discipline

The byte-identical isolated Python environment and offline wheelhouse may be
reused as a dependency closure. The v6 launcher, tracking path, regression
evidence, base lock, and implementation lock are new and disjoint. The
implementation lock must bind all source files, installed distribution RECORDs,
Python, Torch/CUDA/cuDNN, GPUs, deterministic flags, environment variables,
Quanta pins, PIT cache, baseline, and the complete Qlib provider tree.

Mining entitlement is consumed before first v6 label access. All seven mining
topology targets must publish immutably before the single v6 development
opening. All five evaluation arms stage privately; any arm failure cleans every
staging directory and terminalizes the complete evaluation/decision topology.

The v5 MLflow tree is quarantined and used only as hash-bound failure custody.
v6 uses `evaluations/runtime/s2a_v6_tracking` and must not read or append any
predecessor tracking run.

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

- complete v6 authority, custody, runtime, base-lock, and implementation-lock chain;
- immutable KAN, GP/SR, permutation, matched-MLP, mechanism, and blind-review mining artifacts;
- five current Quanta evaluations with exact daily return, turnover, cost, and prediction coverage;
- bootstrap draws, yearly summaries, effective-rank and all integrity criteria;
- one atomically published decision and a human-readable Chinese report;
- terminal evidence, never partial publication, for any failure.
