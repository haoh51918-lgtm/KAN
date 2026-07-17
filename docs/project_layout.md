# Project Layout Contract

All project-owned outputs live under this workspace. External repositories and caches are read-only dependencies referenced by identity.

| Path | Responsibility | May contain generated files? |
|---|---|---|
| `configs/` | Frozen data, model, evaluation, and experiment configuration. | Yes, only reviewed/versioned configs. |
| `prereg/` | Claims, thresholds, budgets, and stop/backtrack rules frozen before results. | Yes. |
| `src/mirage_kan/` | Importable project implementation organized by domain module. | No runtime outputs. |
| `tests/` | Behavior, parity, integration, and regression tests mirroring `src/`. | Test caches are ignored. |
| `factor_libraries/` | Immutable factor-library publications with manifests and panels. | Yes; no checkpoints or logs. |
| `evaluations/` | Evaluation manifests, metrics, predictions, and daily portfolio series. | Yes. |
| `ledgers/` | Append-only candidate, attempt, lineage, and data-access records. | Yes. |
| `artifacts/` | Iteration log, figures, mechanism cards, and other derived research artifacts. | Yes. |
| `experiments/` | Stage trackers and one directory per experiment family. | Yes, but raw logs go to `logs/`. |
| `reports/` | Human-readable checkpoint and final reports; Feishu source reports live in `reports/feishu/`. | Yes. |
| `figures/` | Reproducible figure specifications and, later, publication figures/scripts. | Yes. |
| `audits/` | Read-only source, data, dependency, and result audits. | Yes. |
| `governance/incidents/` | Append-only incident snapshots and custody addenda, hash-bound by the implementation lock. | Yes. |
| `plans/` | Current paper/evidence/execution plans. Superseded plans move to an archive, never masquerade as current. | Yes. |
| `logs/` | Raw command/runtime logs, partitioned by run ID. | Yes. |

No module may write into `src/`, `tests/`, or `configs/` at runtime. Published factor libraries and evaluations are no-replace artifacts. Large external data are referenced by path and SHA-256 rather than copied into the project.

## S1 Gate A matrix layout

Both scientific and smoke matrix roots use the same categorized contract:

| Subdirectory | Responsibility |
|---|---|
| `data/` | One manifest-last copy of raw arrays, indices, masks, and train scaler per seed, saved before training. |
| `checkpoints/` | Validation-selected neural checkpoints partitioned by arm and run ID. |
| `models/` | E2/E3/E4/E5 executable exports, promotion governance, and promoted hard circuits. |
| `residual_shapes/` | Frozen main, null, and source-removed 801-point residual curves published before family fitting. |
| `controls/` | Pre-training control inputs plus post-training null/source-removed evidence manifests. |
| `predictions/` | Matrix and per-arm exclusive claims, row-aligned all-arm predictions, immutable receipts, and final opening manifest per seed. |
| `metrics/` | Immutable pretest summaries, linked accounting, per-seed post-open metrics, bootstrap intervals, and shape curves. |
| `ledgers/` | Training console logs and matrix lifecycle events. |
| `manifests/` | Immutable arm manifests, pretest-ready identity, terminal failure evidence, and the final matrix manifest. A successful matrix manifest contains a complete recursive path/size/SHA-256 index of every other regular file in the run root plus a deterministic aggregate; it does not hash itself. |
| `reports/` | Human-readable smoke or seven-condition Gate A reports without profitability claims. |

Scientific runs live at `artifacts/s1_gate_a_scientific/<matrix_id>/`; reduced
fresh-seed connectivity runs live at `artifacts/s1_gate_a_matrix_smoke/<run_id>/`.
Run IDs must match `[A-Za-z0-9][A-Za-z0-9_-]{0,127}`. Scientific CLI and API
outputs are fixed to the project `artifacts/` root; smoke tests may use isolated
temporary roots.
