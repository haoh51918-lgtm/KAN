# Asset Audit

> Status: initial read-only audit, 2026-07-16 UTC
>
> Historical evidence only. Its authority statement records the rule used on 2026-07-16 and is superseded by root `AGENTS.md`; the verified asset identities remain useful.

## Authority at audit time

The audit was originally conducted under a sole-proposal rule. That rule is no longer current: `KAN_Alpha_PR.md` is now an idea draft, while WIKI remains reference-only unless corroborated.

## Verified QuantaAlpha identity

| Item | Verified evidence | Status |
|---|---|---|
| Source revision | Git commit `b7ceb27b1001261d7a95b209a963664ae1f8ab23` | verified |
| Backtest configuration | `configs/backtest.yaml`, SHA-256 `4e095512025a44dcca279e3d3c4d02fc83367caf044032b6c9f6eeb94405a832` | verified |
| Backtest runner | `quantaalpha/backtest/runner.py`, SHA-256 `a18ec5bfbe57b452dbacb3cdd15249f99c2b53e7c0761c178e9fbb89db7d34d8` | verified |
| Qlib provider | `/zju_0012/htq/aaai26_alpha/QuantaAlpha/data/qlib/cn_data`, approximately 2.9 GB | present |
| Compute | 2 × NVIDIA A800-SXM4-80GB; both idle at audit time | available |

The pinned source config uses CSI300, the one-day label `Ref($close, -2) / Ref($close, -1) - 1`, train 2016–2020, validation 2021, native test/backtest 2022–2025, LightGBM with up to 500 boosting rounds and early stopping 50, and `TopkDropoutStrategy(topk=50, n_drop=5)` with the configured open-price transaction-cost model.

## Verified raw cache identity and data checks

Cache: `/zju_0012/htq/aaai26_alpha/05_experiments/_cache/pit_full_csi300_qa_v2.parquet`

| Check | Result |
|---|---:|
| Cache SHA-256 | `cbcf1c0e06f0a966f503d9c2fc1688fbe78faee5a9a46b99f32abf9498229f69` |
| Rows | 1,572,483 |
| Dates | 2,671, from 2015-01-05 to 2025-12-26 |
| Instruments | 653 |
| Duplicate instrument-date rows | 0 |
| Dynamic-universe rows | 801,001 |
| Members per day, min / median / max | 298 / 300 / 300 |
| One-day label recomputation | exact; maximum absolute difference 0 |
| Twenty-day label recomputation | exact; maximum absolute difference 0 |
| In-universe rows with all raw OHLCV fields non-finite | 15,394 |

The cache, Qlib calendar, CSI300 membership file, and all-instruments file match the SHA-256 identities in the cache manifest. Universe membership and observed/finite masks must remain separate because some valid membership rows have no raw observation.

## Historical implementation assets

Focused tests were run from `/zju_0012/htq/aaai26_alpha` with the system Python 3.12 interpreter.

| Asset family | Evidence | Reuse decision |
|---|---|---|
| PIT loader and mask semantics | included in passing focused suite | reference or narrow verified adaptation |
| Typed AST registry and Pandas/Torch parity | included in passing focused suite | reference only; proposal-specific semantics require a new identity |
| Whole-program beam/refit kernel | included in passing focused suite | reuse algorithmic ideas behind a new interface |
| Joint-support LightGBM comparison kernel | included in passing focused suite | adapt to pinned QuantaAlpha 500-round/ES50 protocol; do not inherit fixed14 |
| Atomic/source-identity artifact helpers | included in passing focused suite | narrow reuse is acceptable |
| Official pykan numerical wrapper | test collection blocked because the current system environment lacks `scikit-learn` | unresolved until project environment is created |

Test result excluding the environment-blocked pykan wrapper: **77 tests and 102 subtests passed** in 11.66 seconds.

## Historical results

The existing Alpha158 result at `/zju_0012/htq/aaai26_alpha/09_kan_factor/results/metrics_baseline_a158lgb_qa_FAITHFUL.json` is a hashable baseline lead and may avoid unnecessary reproduction. It is not a MIRAGE result. Historical fixed14 evaluations are not comparable to the pinned QuantaAlpha 500-round/early-stopping protocol and must not share a result row.

## Rejected inheritance

- WIKI-specific proposal variants, gates, trackers, or claim ceilings as method authority.
- Old runner state, preregistration identity, or route-specific status.
- Old operator semantics under a new name without new behavior tests.
- Neural scores, selected pre-existing formulas, or curve plots presented as a mined canonical factor library.
- Any historical metric promoted without its source, configuration, data role, signal object, and artifact identity.
