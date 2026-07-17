# MIRAGE-KAN

This workspace investigates whether typed KAN-based search can produce interpretable, independently executable alpha-factor libraries with useful cost-aware QuantaAlpha performance.

Current behavior and authority are defined by `AGENTS.md`; current scientific state is defined by `docs/research/MIRAGE_KAN_LIVING_MANUAL.md`. `KAN_Alpha_PR.md` is the original idea draft and hypothesis catalogue, not an execution contract. Frozen preregistrations, openings, incidents, and result artifacts remain immutable evidence for the runs they own.

## Current status

See `docs/research/MIRAGE_KAN_LIVING_MANUAL.md` for the single current evidence boundary and `plans/todos.md` for the single current action queue. In brief: S1b is the next clean experiment; S2a v8 is closed/quarantined; no v9 is authorized.

The root-cause and remediation audit is in `docs/research/PROJECT_AUDIT_2026-07-17.md`.

## Evidence boundaries

Scientific openings and frozen result files are irreversible. Ordinary software fixes, tests, documentation, dry runs, exact enumeration, and no-label loader checks are reversible engineering and must not trigger a new scientific protocol by default. An implementation failure is debugged red-first on the same reversible path; a new preregistration is required only when a hypothesis, data role/opening, seed, threshold, or statistical decision rule changes.

2022-2025 has already been used as development evidence. The 2026 H1 confirmation period remains closed until the retained method and confirmation protocol are explicitly frozen.

## Key artifacts

| Path | Contents |
|---|---|
| `AGENTS.md` | Current agent behavior and authority contract |
| `docs/research/MIRAGE_KAN_LIVING_MANUAL.md` | Single current scientific-state source |
| `plans/todos.md` | Single current action queue |
| `docs/research/PROJECT_AUDIT_2026-07-17.md` | Root-cause, plugin, document-conflict, and prevention audit |
| `audits/s0_real_data_audit.json` | Cache identity, mask counts, and exact label-value parity |
| `factor_libraries/seed_ast_v1/` | S0 wiring-control AST library and panel |
| `evaluations/s0_vertical_slice/` | S0 real Quanta connectivity evaluation |
| `artifacts/s1_gate_a_scientific/s1_gate_a_v1_scientific_formal_v1/` | Sealed Gate A evidence |
| `reports/s1_gate_a_scientific_report.md` | Gate A result and evidence boundary |
| `prereg/s1b_identifiability_v1.md` | Unexecuted S1b scientific plan |
| `prereg/s2a_kan_e3_vertical_v8.md` | Historical v8 protocol; closed, not current instructions |
| `evaluations/s2a_kan_e3_vertical_v8_decision/terminal_failure.json` | v8 terminal decision-assembly failure |

## Verification environment

Use the project runtime, not `/usr/bin/python`:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
MLFLOW_ALLOW_FILE_STORE=true \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONHASHSEED=0 \
QLIB_DATA_DIR=/zju_0012/htq/aaai26_alpha/QuantaAlpha/data/qlib/cn_data \
PYTHONPATH="$PWD/src" \
runtime/s2a_v4_eval/.venv/bin/python -m pytest -q

PYTHONPATH="$PWD/src" \
runtime/s2a_v4_eval/.venv/bin/python -m mirage_kan.cli \
  --workspace . verify-library --library factor_libraries/seed_ast_v1
```

The system Python lacks required project dependencies and is not a valid signal for the repository test status.

## Run a fresh non-scientific S0 chain

All artifact destinations are no-replace, so choose new run names:

```bash
MLFLOW_ALLOW_FILE_STORE=true \
QLIB_DATA_DIR=/zju_0012/htq/aaai26_alpha/QuantaAlpha/data/qlib/cn_data \
PYTHONPATH=src \
/zju_0012/htq/aaai26_alpha/09_kan_factor/fullform/.venv_qlib/bin/python \
  -m mirage_kan.cli --workspace . run-s0 \
  --audit-output audits/s0_real_data_audit_<run>.json \
  --library factor_libraries/seed_ast_<run> \
  --evaluation evaluations/s0_vertical_slice_<run>
```

This is an engineering/connectivity run unless a separate prospective scientific protocol says otherwise. Never reuse a consumed run ID or artifact destination.
