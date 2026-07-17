# Incident: S2a v3 strict-library admission failure

- **Date**: 2026-07-17 UTC
- **Protocol**: `s2a_kan_e3_vertical_v3`
- **Classification**: `scientific_screen_failure_before_development`
- **Failure boundary**: production KAN hard-AST library admission
- **Development opened**: no
- **Quanta executed**: no

## Outcome

The complete v3 mining computation ran through real KAN, within-date label-
permutation KAN, and typed GP/SR candidate generation and exact hard-AST
scoring. Production selection admitted seven strict KAN factors. The frozen
minimum library size was eight, so `require_complete()` failed before the MLP
control, mechanism-card publication, development opening, or Quanta backtest.

The profile quota passed. The failure was therefore library-count insufficiency
under the frozen admission rule, not a profile-collapse failure and not an
infrastructure failure.

## Evidence boundary

The only observed scientific summary is the fail-closed exception:

`hard-AST selection is incomplete: selected=7, target=None, minimum_size_met=False, profile_quota_met=True`

No candidate score, factor identity, development outcome, portfolio metric, or
Quanta result was published. All seven v3 topology targets terminalized and
must not be deleted, resumed, or relabeled as successful artifacts.

## Custody pins

| Evidence | SHA-256 |
|---|---|
| v3 base lock | `07faa9b04368ce757e032def7779a22e7fdf42cf30bf340f24900a4f94bb3567` |
| v3 implementation lock | `cce2660f56eb189f2c6545b7314d931b30080376d16d15a5f5bf62d2a82ff235` |
| mining entitlement | `e65e48db4cce3b1b28d4025279cfa144b0e6e666410ed369268b90e936861ac3` |
| mining preclaim | `498584e8089139b60a81c69791a2b026a7015a0c691c5b33212730e224ca5931` |
| authority ledger | `ce80260e7d70c2b5e6b58e6f1f2eb7d9f01b0ab14562e65004c7241e42377221` |
| mining terminal receipt | `4738dccd61a1141c699c6d11a5adab348225bc606ad907bb2babc17e6b42697a` |
| KAN library terminal receipt | `25a2b1909b94c24270395baca407faf6868023a3f6e86f7a23016c3cef770a4e` |
| GP/SR terminal receipt | `44d2d2759817b83443753b6929d8d3cd27d61820fc8af997427fb141fb10f2a9` |
| permutation terminal receipt | `55ae3562173f272ba0ffa5482bb83cfe4dfe913580ad893c28ebeaff1445cf16` |
| MLP terminal receipt | `4e511033bca9fa153ecccb78556b9f695071bd9eaaa697f79246156ed8712309` |
| mechanism-card terminal receipt | `fc67b5dd557c2ba77b95789531cc6686b122ca103857a78fcd8547f09b4156ae` |
| blind-package terminal receipt | `35049d8051bb9b8ac20efc93b93ef1a4f37cb4ff0d0883c62a4c277c496222df` |

## Adaptive successor rule

The user-defined primary objective is factor-library backtest quality, not the
number of factors. A count-only gate must not indefinitely prevent the first
complete backtest when seven factors already pass all per-factor quality and
diversity rules. The successor will therefore:

1. keep all per-factor RankIC, coverage, sign, fidelity, gate-margin, diversity,
   profile, seed, training, and control thresholds unchanged;
2. set the exploratory developmental minimum library size and corresponding
   integrity minimum to six, not to the observed count seven, so the rule is not
   fitted exactly to this result;
3. keep the library cap at 16 and the minimum profile count at three;
4. remain developmental and forbid formal promotion;
5. rerun the complete mining/control computation under a new protocol because
   v3 published no reusable factor library; and
6. include the complete QLib/LightGBM/MLflow/Torch evaluation environment in
   the new implementation closure before any label or development access.

