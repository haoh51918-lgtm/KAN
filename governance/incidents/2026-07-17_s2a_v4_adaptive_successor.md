# Governance authorization: S2a v4 adaptive complete-chain successor

- **Date**: 2026-07-17 UTC
- **Predecessor**: `s2a_kan_e3_vertical_v3`
- **Successor**: `s2a_kan_e3_vertical_v4`
- **Authorization class**: adaptive developmental successor after count-gate failure
- **Development-period observation in predecessor**: none
- **Formal promotion**: forbidden

## Why a successor is authorized

v3 completed its real-label mining/search/scoring computation but selected seven
strict KAN factors against a frozen minimum library size of eight. The profile
quota passed. The run terminated before the MLP control, all scientific artifact
publication, development opening, and Quanta backtest.

The project's user-defined primary horizontal criterion is factor-library
backtest quality under the fixed Quanta framework. Library count is secondary.
The v3 count gate therefore prevented the primary question from being tested by
one factor even though every retained factor passed the unchanged quality rules.

v4 is authorized to use a six-factor exploratory minimum, not the observed
count seven. Six gives two factors per required profile on average while
retaining redundancy and avoiding an exact post-hoc fit. The library cap remains
16 and at least three profiles remain mandatory.

## Changes permitted in v4

1. Protocol identity and every writable path move to v4.
2. `admission.minimum_library_size` changes from eight to six.
3. `s2a_decision.integrity.production_library_size_minimum` changes from eight
   to six so selection and decision semantics agree.
4. The implementation closure expands to the complete isolated evaluation
   runtime, including every installed distribution and its `RECORD` hash,
   wheel/lock manifests, Python, Torch/CUDA/cuDNN, QLib, LightGBM, MLflow,
   Quanta, provider, and required environment variables.
5. Terminal failures may record aggregate rejection diagnostics, but partial
   candidate membership remains non-publishable and non-reusable.

No other scientific configuration may change. In particular, KAN/GP/
permutation budgets, seeds, train/validation/development periods, label purge,
raw-only warm-up, atom space, optimizer, admission efficacy thresholds,
fidelity, diversity, mechanism evidence, Quanta costs, bootstrap, guardrails,
and performance thresholds remain identical to v3.

## v3 custody pins

| Evidence | SHA-256 |
|---|---|
| v3 base lock | `07faa9b04368ce757e032def7779a22e7fdf42cf30bf340f24900a4f94bb3567` |
| v3 implementation lock | `cce2660f56eb189f2c6545b7314d931b30080376d16d15a5f5bf62d2a82ff235` |
| v3 admission-failure incident | `a9eac2d395754570d114f14b31d94ffd9029ceafae79412bbc6d3240fd70ab82` |
| v3 mining entitlement | `e65e48db4cce3b1b28d4025279cfa144b0e6e666410ed369268b90e936861ac3` |
| v3 mining preclaim | `498584e8089139b60a81c69791a2b026a7015a0c691c5b33212730e224ca5931` |
| v3 authority ledger | `ce80260e7d70c2b5e6b58e6f1f2eb7d9f01b0ab14562e65004c7241e42377221` |
| v3 mining terminal | `4738dccd61a1141c699c6d11a5adab348225bc606ad907bb2babc17e6b42697a` |
| v3 KAN-library terminal | `25a2b1909b94c24270395baca407faf6868023a3f6e86f7a23016c3cef770a4e` |
| v3 GP/SR terminal | `44d2d2759817b83443753b6929d8d3cd27d61820fc8af997427fb141fb10f2a9` |
| v3 permutation terminal | `55ae3562173f272ba0ffa5482bb83cfe4dfe913580ad893c28ebeaff1445cf16` |
| v3 MLP terminal | `4e511033bca9fa153ecccb78556b9f695071bd9eaaa697f79246156ed8712309` |
| v3 mechanism-card terminal | `fc67b5dd557c2ba77b95789531cc6686b122ca103857a78fcd8547f09b4156ae` |
| v3 blind-package terminal | `35049d8051bb9b8ac20efc93b93ef1a4f37cb4ff0d0883c62a4c277c496222df` |

The four v3 authority receipts and their no-replace claim files are also direct
custody inputs to the v4 base/implementation lock. All v2 custody files remain
bound transitively and should remain direct v4 inputs where practical.

## Required gates before v4 label access

1. v3 evidence above remains byte-identical.
2. Parsed v4 configuration differs from v3 only in protocol identity, writable
   paths, and the two six-factor minimum fields.
3. v4 paths are disjoint from v2 and v3 paths.
4. The isolated runtime passes offline sync, full dependency check, complete
   import smoke, and distribution-inventory verification.
5. The complete test and Ruff suites pass under the v4 environment.
6. A new v4 implementation lock is issued once and reverified.

