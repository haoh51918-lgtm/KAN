# S2a v7 pre-opening loader correction and exact mining rebind

- **Date**: 2026-07-17 UTC
- **Predecessor**: `s2a_kan_e3_vertical_v6`
- **Successor**: `s2a_kan_e3_vertical_v7`
- **Evidence class**: corrective adaptive repeated-development screen
- **Development labels accessed by v6**: no

## Incident

v6 completed and atomically published its full seven-part mining topology. Two
independent read-only audits passed. Before consuming the v6 development
opening, a newly adopted real no-label full-chain rehearsal exercised the
matched-MLP loader on the locked sparse PIT panel. The publication path had
projected a Cartesian atom-panel replay back to the raw PIT index, while the
checkpoint verifier compared the unprojected Cartesian replay directly with
the published sparse prediction parquet. The resulting shape mismatch was
964,314 versus 868,920 rows.

The failure is an evaluation-loader index-semantics defect. It is not a mining,
selection, parameter, checkpoint, prediction, lineage, or artifact-integrity
failure. A read-only simulation applying the same exact raw-index projection as
publication verified all seven MLP controls and the complete no-label panel.

## Disposition

v6 is frozen with successful mining and an unconsumed development opening. Its
implementation identity is not changed. v7 applies the one-line exact replay
projection under a new implementation lock and references the entire v6 mining
topology through `mirage_mining_rebind_receipt_v1`.

The rebind is byte-exact whole-topology custody. It does not create a v7 mining
preclaim or label entitlement, copy or mutate source artifacts, select or reorder
factors, change thresholds, retune parameters, or reuse any v5 development
metric. Every source payload is rehashed before the v7 development opening and
again during live verification.

## Scientific continuity

Train, validation, repeated-development dates, labels, purge, raw-only warm-up,
search space, KAN and control budgets, seeds, admission, interpretability,
Quanta evaluation, portfolio costs, metrics, bootstrap, guardrails, and decision
thresholds remain equal to v6. The previously authorized graph unlock becomes
prospective in v7 governance, but graph is not used to alter this fixed v6
factor library or the current five-arm evaluation.

## Required pre-opening gates

1. red-first ragged-PIT integration test and exact value-mismatch negative test;
2. deterministic, exclusive, fail-closed cross-protocol rebind receipt;
3. v7 base and implementation locks;
4. full test and Ruff regression;
5. complete synthetic shadow rehearsal of opening, five arms, and decision;
6. real no-label AST and MLP loader replay;
7. two independent read-only audits.

Any failure before the development opening preserves the unopened data boundary
and requires another prospective successor rather than an in-place mutation.
