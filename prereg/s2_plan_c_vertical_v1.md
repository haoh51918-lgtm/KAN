# S2a Plan C heterogeneous factor-library preregistration

## Status and claim boundary

This protocol is prospective and must be locked before any S2 candidate score or
2022–2025 S2 portfolio result is read. `KAN_Alpha_PR.md` is the sole proposal
authority. WIKI and predecessor results are not used to choose formulas or
thresholds.

S2a is the smallest faithful end-to-end development screen after Gate A failed.
It can authorize a larger S2 formal matrix, but it cannot establish final factor-
library value, out-of-sample superiority, or a paper claim. The 2022–2025 period
is development evidence. Graph control remains locked regardless of the S2a
outcome.

## Question

Can a non-graph heterogeneous set of typed symbolic miners produce an immutable,
independently executable factor library whose real pinned Quanta portfolio result
is materially better than Alpha158 and matched random/permuted controls without
materially worsening drawdown, RankIC, turnover, or trading cost?

## Complete-chain priority

S2a deliberately uses one pinned LightGBM seed and a 256-candidate search so the
entire candidate-generation → screening → immutable-library → Quanta →
portfolio-decision path runs before any miner is optimized. The larger five-seed,
three-fold, nineteen-permutation design is prospectively specified as an
escalation and runs only if S2a passes every signal and guardrail.

## Data access

- Discovery train: 2016-01-01 through 2020-12-31.
- Candidate validation: 2021-01-01 through 2021-12-31.
- Development portfolio test: 2022-01-01 through 2025-12-26.
- Label: the verified one-day `fwd`, identical to Quanta's
  `Ref($close,-2)/Ref($close,-1)-1`.
- All selected, random, and permutation libraries must be immutable before the
  first S2 development-test evaluation.
- Formula selection may inspect train/validation labels only. The development
  portfolio period is opened once for the four predeclared arms.

Membership, raw observation, expression support, finite output, label availability,
and tradability remain distinct. Candidate coverage uses membership, finite label,
and expression support; missing raw values are never imputed by the miner.

## Miners and budgets

Four non-graph typed symbolic profiles are used: trend, mean reversion,
price-volume, and typed composition. They differ by seeds, templates, window
priors, and legal composition rules but share one DSL and executor. Each receives
64 attempted canonical programs, for exactly 256 attempts and at most 256 full
development evaluations. Invalid, duplicate, numerically invalid, and successful
attempts all consume budget and enter the ledger.

KAN has no mandatory quota because Gate A produced no governed HARD primitive.
This is Plan C behavior, not an approximation of a KAN miner.

Programs use only the frozen OHLCV leaves, current legal typed operators, windows
in 2/3/5/10/20/40/60, maximum depth six, and maximum 20 AST nodes. The selected
library cap is 16, the minimum valid size is eight, at least three profiles must be
represented, and padding is forbidden.

## Candidate scoring and admission

For each date, RankIC is the Spearman correlation across supported in-universe
assets. A candidate is eligible only when:

1. type, causality, domain, support, and finite-output checks pass;
2. its canonical hash is unique;
3. development coverage is at least 0.85;
4. absolute mean daily RankIC is at least 0.005 in train and validation;
5. train and validation RankIC signs agree; and
6. its absolute validation Spearman correlation with every already selected
   factor is below 0.80.

Eligible candidates are ordered by the smaller of absolute train and validation
RankIC, descending; ties use fewer AST nodes and then canonical hash. Greedy
selection enforces at least one factor from three miner profiles before filling
remaining capacity. Formula polarity is never changed from validation or test
results; LightGBM may learn either sign normally.

## Direct controls

- `alpha158_replay`: one real current replay of the hash-bound historical anchor.
- `heterogeneous_selected`: the label-scored library.
- `random_typed`: a label-free, deterministic sample from type-, support-, and
  finite-valid candidates with the same library cap.
- `label_permutation_selected`: reruns scoring and admission after a deterministic
  within-date label permutation, without reusing the observed selected library.

All custom libraries use the same publication code, panel index, Quanta adapter,
LightGBM 500/ES50, TopkDropout 50/5, open execution, and cost model. The replay is
accepted only when its IR, MDD, and RankIC differ from the frozen historical anchor
by no more than 0.03, 0.02, and 0.003 respectively. A replay miss is infrastructure
inconclusiveness, not a negative scientific result.

## S2a decision

`advance_s2_formal` requires every condition below:

| Condition | Frozen threshold |
|---|---:|
| Selected net Information Ratio | at least 0.32 |
| Selected minus replay IR | at least +0.10 |
| Selected minus random IR | at least +0.05 |
| Selected minus permutation IR | at least +0.05 |
| Selected minus replay maximum drawdown | at least -0.03 |
| Selected minus replay RankIC | at least -0.003 |
| Selected/replay mean turnover ratio | at most 1.20 |
| Selected/replay mean realized cost ratio | at most 1.20 |
| Calendar years with selected active return not below replay | at least 3 of 4 |

Information Ratio (IR) is the annualized mean divided by volatility of daily net
benchmark-relative return. RankIC is daily cross-sectional rank correlation.
Turnover and realized cost are captured from the exact Qlib portfolio report used
by the pinned Quanta run, not estimated by a proxy.

If the replay is valid but any signal or guardrail fails, the result is
`s2a_screen_fail`. If integrity, identity, or replay validation fails before a
scientific comparison, the result is `s2a_inconclusive_infrastructure`.
S2a never emits `PROMOTE_LIBRARY_VALUE`.

## Formal escalation

Only `advance_s2_formal` unlocks the preregistered expansion: model seeds
42/1042/2042/3042/4042, three rolling candidate folds, 19 full-process label
permutations, and a 10,000-replicate 20-trading-day moving-block bootstrap. The
formal primary comparison is paired against Alpha158 and requires the one-sided
95% lower bound of delta IR above zero. Exact formal replacement/incremental arms
will be locked in a separate v1-formal implementation manifest without changing
the S2a result.

## Required artifacts

- append-only attempt and data-access ledgers;
- every attempted canonical AST and disposition;
- train/validation candidate metrics and selection order;
- immutable selected, random, and permutation factor libraries;
- current Alpha158 replay plus three custom Quanta evaluations;
- daily excess return, turnover, cost, prediction coverage, metrics, console logs,
  source/config identities, and a complete no-replace top-level manifest;
- a machine decision and a human-readable Chinese report.

