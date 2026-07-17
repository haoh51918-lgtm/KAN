# Governance authorization: S2a v5 calendar-corrective successor

- **Date**: 2026-07-17 UTC
- **Predecessor**: `s2a_kan_e3_vertical_v4`
- **Successor**: `s2a_kan_e3_vertical_v5`
- **Authorization class**: corrective adaptive successor after development infrastructure failure
- **Predecessor observation**: inconclusive infrastructure failure with quarantined development outputs
- **Prospective one-shot status**: spent; v5 is repeated developmental evidence
- **Formal promotion**: forbidden

## Failure disposition

v4 completed and immutably published its seven-part mining topology. It then
consumed the one development opening and ran Alpha158 followed by the selected
KAN arm. Before any arm publication, the selected-KAN staging bundle failed the
exact prediction-calendar check. The development transaction removed all arm
staging and terminalized five evaluation targets plus the decision target. No
development manifest or decision was published.

The exact failure is not a missing-date approximation. The frozen development
calendar contains 966 trading dates from 2022-01-04 through 2025-12-26.
Alpha158 predictions contained exactly those 966 dates. The selected-KAN
prediction signal contained the same 966 dates plus 1,461 train/validation
dates, for 2,427 dates from 2016-01-04 through 2025-12-26.

The pinned Quanta YAML represents `train`, `valid`, and `test` ranges as lists.
Its precomputed-factor `PrecomputedDataHandler.fetch` recognizes only tuple or
slice date selectors. All three custom-factor selectors were therefore
silently ignored, so train, validation, and test each returned the full
2016--2025 panel. Development labels entered model fitting. Every v4 custom-arm
metric, prediction, label, model diagnostic, and portfolio statistic is invalid
and quarantined. Cropping the signal, reindexing coverage, or relaxing the
calendar gate is forbidden because none would repair the contaminated fit.

## Permitted v5 changes

1. Move the protocol identity and every writable mining, evaluation, decision,
   authority, opening, recovery, report, and tracking path to v5.
2. In `QuantaAdapter.evaluate_panel` only, validate the three frozen two-date
   sequences and temporarily normalize the YAML lists to equal-value tuples
   while the pinned precomputed dataset is constructed. Restore the original
   runner config in `finally`.
3. Record the exact unchanged segment values in the Quanta execution identity.
4. Admit the precise predecessor-custody class
   `inconclusive_infrastructure_with_quarantined_development_outputs` while
   continuing to reject generic or unclassified development observation.
5. Label v5 outputs `corrective_adaptive_repeated_development_screen`; never
   represent them as a fresh prospective one-shot or final confirmation.
6. Reuse the byte-identical isolated dependency environment and wheelhouse, but
   use a new v5 launcher, tracking directory, locks, and runtime evidence.

No scientific field may change. Search space, 256/256/256 attempt budgets,
seeds, optimizer, 300 updates, admission thresholds, six-factor minimum,
16-factor cap, profile quota, controls, training/validation/development dates,
label purge, warm-up, Quanta commit/config/runner, LightGBM settings, portfolio,
costs, metrics, bootstrap, guardrails, and decision thresholds remain byte-
equivalent after removing protocol identity, writable paths, and evidence-class
metadata.

Alpha158 must continue through the unchanged official runner call. The adapter
normalization must not crop prediction signals or special-case coverage.

## Quarantine and custody

The following v4 material is predecessor evidence only:

- v4 config, preregistration, base lock, implementation lock, and runtime lock;
- mining preclaim, entitlement, top bundle, six child bundles, and all payloads;
- development preclaim and consumed opening;
- 15 authority receipts, claims, ledger, and two arm-consumption records;
- six development terminal-failure records;
- both MLflow run trees, including predictions, labels, metrics, and analysis;
- implementation-lock and mining reports plus the development blocker report.

The tracking tree is retained for custody and failure diagnosis, not for v5
thresholds, seeds, method choices, early stopping choices, or scientific
narrative. The already observed invalid custom-arm values cannot be cited as
evidence. v4 mining members are also not copied into active v5 topology; v5
reruns the complete deterministic mining chain under its own identities because
cross-protocol topology reuse would require a broader and riskier mechanism.

## Required gates before v5 label access

1. A synthetic test reproduces list-selector leakage and passes only when the
   actual pinned dataset prepares disjoint train, valid, and test periods.
2. The original runner config is restored after dataset construction, and the
   Alpha158 official path remains unchanged.
3. Exact coverage validation remains enabled and requires the 966-date calendar.
4. Full tests, Ruff, offline dependency verification, import audit, Quanta pins,
   provider identity, GPU identity, and deterministic state all pass.
5. The v5 base lock binds the complete v4 quarantine and the v5 runtime closure.
6. A new v5 implementation lock is written exclusively and verified live before
   mining entitlement consumption.

Even if v5 passes every autonomous gate, it is corrective developmental
evidence. Formal promotion still requires human blind review and genuinely
unseen confirmation evidence.
