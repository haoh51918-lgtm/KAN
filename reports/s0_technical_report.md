# S0 Real Vertical Slice — Technical Handoff

> Status: completed on 2026-07-16 UTC  
> Evidence class: wiring control only; not KAN-mined and not a MIRAGE scientific result

## Outcome

The project-owned path now runs from the pinned PIT OHLCV cache through a typed canonical AST, immutable factor-library publication, QuantaAlpha's real precomputed-factor dataset path, LightGBM with a 500-round cap and early stopping 50, and the real cost-aware TopkDropout backtest.

| Check | Result |
|---|---:|
| Unit/behavior tests | 11 passed |
| PIT rows | 1,572,483 |
| Dynamic-universe rows | 801,001 |
| In-universe rows with all OHLCV unobserved | 15,394, preserved |
| 1-day label comparable rows / max absolute difference | 784,283 / 0 |
| 20-day label comparable rows / max absolute difference | 773,159 / 0 |
| Published seed factors | 4 |
| Independent library recomputation | passed |
| Real backtest steps | 966 |

The label missing mask is intentionally not equated with raw-close availability. Recorded labels are dynamic-universe scoped, while raw close covers a wider instrument panel; right-boundary labels can also use Qlib dates beyond the cache end. No recorded finite label occurs outside the dynamic universe.

## Wiring-control observations

| Metric | Value |
|---|---:|
| IC | 0.042781 |
| ICIR | 0.261628 |
| RankIC | 0.041031 |
| RankICIR | 0.254987 |
| Net information ratio | 0.528703 |
| Net annualized excess return | 0.042857 |
| Maximum drawdown | -0.155985 |

These values demonstrate evaluator connectivity only. The library is hand-wired from proposal-legal ASTs, explicitly has `kan_mined=false`, and cannot support a MIRAGE-KAN improvement claim.

## Reproduction commands

Run from the workspace root with fresh no-replace destination names:

```bash
PYTHONPATH=src /usr/bin/python -m pytest -x -q --tb=short
PYTHONPATH=src /usr/bin/python -m mirage_kan.cli --workspace . audit-data --output audits/s0_real_data_audit_<run>.json
PYTHONPATH=src /usr/bin/python -m mirage_kan.cli --workspace . publish-seed --destination factor_libraries/seed_ast_<run>
MLFLOW_ALLOW_FILE_STORE=true QLIB_DATA_DIR=/zju_0012/htq/aaai26_alpha/QuantaAlpha/data/qlib/cn_data PYTHONPATH=src /zju_0012/htq/aaai26_alpha/09_kan_factor/fullform/.venv_qlib/bin/python -m mirage_kan.cli --workspace . evaluate --library factor_libraries/seed_ast_<run> --destination evaluations/s0_vertical_slice_<run>
```

The MLflow environment switch is required by the installed MLflow release to permit Qlib's filesystem tracking backend. It does not change the model, data, portfolio, or cost configuration.

## Published artifacts

- Data audit: `audits/s0_real_data_audit.json`
- Factor library: `factor_libraries/seed_ast_v1/`
- Evaluation: `evaluations/s0_vertical_slice/`
- Candidate ledger: `ledgers/attempts.jsonl`
- Iteration history: `artifacts/iteration_log.md`

The factor library and evaluation use GPFS-compatible exclusive-directory, exclusive-file, manifest-last, fsync publication. Existing destinations are never replaced.
