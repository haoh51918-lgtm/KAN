# MIRAGE-KAN Paper and Evidence Plan

> Status: working evidence plan, updated 2026-07-17 UTC  
> Current authority/state: `AGENTS.md` and `docs/research/MIRAGE_KAN_LIVING_MANUAL.md`  
> Original idea draft: `KAN_Alpha_PR.md`  
> Historical WIKI/code: untrusted reference unless independently verified

## 1. Smallest defensible claim

No positive method claim has yet been earned. The next claim we try to earn is deliberately narrow:

> A typed KAN-based search can produce independently executable factors whose response behavior is understood by blind reviewers and whose library is not materially worse than simpler matched alternatives under a prospectively frozen QuantaAlpha comparison.

The graph is a conditional hypothesis, not a locked destination. Reversible interface work may proceed, but a graph claim requires a varied candidate pool and superiority to same-information flat and bandit controllers.

## 2. Story design

| Element | MIRAGE-KAN statement |
|---|---|
| Task | Discover auditable alpha programs whose library improves an actual downstream portfolio, not merely an isolated IC score. |
| Challenge | Alpha search is low-signal and non-stationary; independent generators repeatedly rediscover correlated candidates, while unconstrained KAN outputs are hard to export as faithful formulas. |
| Insight | Continuous function fitting and discrete program construction should be separated but trained together; the discovered factor population can then provide feedback that changes what miners generate next. |
| Method | Typed Symbolic-Residual KAN miners produce canonical programs; a Miner–Factor controller assigns residual tasks, diversity pressure, typed evolution, and budget. |
| Advantage to test | Better cost-aware factor-library backtests under matched support and budget, with lower redundancy and standalone executable formulas. |

### Module motivation map

| Module | What it does | Why it is needed | Evidence required |
|---|---|---|---|
| Typed financial DSL | Restricts leaves, operators, windows, types, causality, domains, and mask propagation. | Rejects invalid or leaky programs before expensive evaluation. | Invalid-program tests, future-access tests, Pandas/Torch parity, unique-valid yield. |
| Symbolic-Residual KAN | Combines analytic primitive gates with a penalized spline residual and explicit hardening. | Pure symbolic dictionaries can miss useful shapes; free splines do not directly produce faithful formulas. | E1–E5 synthetic recovery, soft-to-hard fidelity, complexity–quality Pareto. |
| Factor-library evaluator | Publishes canonical programs and evaluates them with the pinned Quanta stack. | The research target is portfolio value of a factor library, not miner loss. | Hash-bound library, joint support, replacement/incremental backtests. |
| Miner–Factor feedback | Uses discovered factors to route residual tasks and diversity pressure. | Independent miners may repeatedly explore the same behavior. | Generation-shift metrics versus Independent, Boost-Sequential, FactorGraph-Select, and Flat-Controller. |
| Typed evolution and budget | Changes program structure and allocates evaluation effort. | Search must improve useful library yield under finite resources. | Matched-budget comparison versus random evolution and Bandit-Budget. |

## 3. Pre-mortem: likely rejection comments

| Priority | Predicted rejection | Decisive preemption | Claim response if it fails |
|---:|---|---|---|
| 1 | “KAN is branding; a symbolic searcher or MLP is enough.” | Execute the already-preregistered S1b mechanism battery, blind interpretability pilot, and matched simpler controls. | Report the boundary honestly; retain KAN only for claims the evidence supports. |
| 2 | “Any gain comes from LightGBM, factor count, or different sample support.” | Same joint support, common library cap, same 500-round/ES50 LightGBM, replacement and incremental arms. | Do not claim factor-library quality. Repair the comparison contract first. |
| 3 | “The graph is boosting/adaptive allocation with extra machinery.” | Independent, Boost-Sequential, Bandit-Budget, and same-information Flat-Controller at matched candidate/full-evaluation/compute budgets. | Reframe as non-graph closed-loop control or stop at the single-miner paper. |
| 4 | “The result is leakage or winner’s curse.” | Separate membership/observed/tradable masks, purged pseudo-future search, complete attempt ledger, whole-pipeline label permutation, HAC/block-bootstrap uncertainty. | Stop scientific interpretation and return to the data/evaluation contract. |
| 5 | “The formulas are readable but not faithful explanations.” | Soft/hard agreement, variable deletion, lag masking, learned-shape comparison, and cross-seed structure stability. | Label outputs semi-symbolic or neural and remove mechanism claims. |

## 4. Evidence dependency stages

These stages are not a restatement of proposal Phase 0–6. They are ordered by what evidence unlocks the next expensive decision.

| Stage | Outcome that unlocks the next stage | Stop/backtrack trigger |
|---|---|---|
| S0 — real vertical slice | PIT OHLCV → typed canonical AST → immutable factor library → pinned Quanta LightGBM → real TopkDropout backtest runs end to end. | Any identity, label, mask, causal, support, or adapter parity failure returns to S0. |
| S1 — KAN mechanism test | Gate A v1 is already a valid negative result; S1b tests whether an identifiable variant recovers registered mechanisms and exports standalone programs. | Preserve either outcome; no automatic S1c or threshold repair. |
| S2 — search-space and library value | Exact-space/pool diagnostics justify the search space, then a future frozen library comparison tests cost-aware value. | Expand or abandon the weak space based on diagnostics; do not create S2a v9 or use quarantined v8 metrics. |
| S3 — independent-miner diagnosis | Multiple miners preserve S2 quality but exhibit material structural/behavioral redundancy not solved by simple heterogeneity. | No redundancy means there is no demonstrated need for graph coordination. |
| S4 — minimal graph feedback | Residual-task routing plus diversity repulsion changes the generation distribution and beats non-graph adaptive controls. | Graph not beating Flat-Controller/Boost-Sequential removes the graph claim. |
| S5 — MIRAGE closed loop | Typed evolution and budget control add factor-library backtest value under matched search and compute budgets. | A failing path is removed; no bundled module receives credit. |
| S6 — robustness and confirmation | Multi-seed, rolling, negative-control, decay, and intervention evidence supports the retained claims. | 2022–2025 remains development evidence; an untouched lockbox is required for a final confirmation claim. |

S0 and the synthetic part of S1 may run in parallel once the DSL contract exists. S1 candidates must enter the same S0 publication/evaluation path; no synthetic-only surrogate can pass S2.

## 5. Claim-to-experiment map

| Claim | Primary evidence | Status | Promotion rule |
|---|---|---|---|
| C1: exported factor programs are causal and independently executable | S0 contract tests and independent recomputation | interface supported | Exact identity, mask, causality, and recomputation checks pass. |
| C2: an identifiable KAN contributes beyond direct controls | S1b mechanism battery | untested | The preregistered recovery/fidelity rule passes. |
| C3: mined factors improve a real downstream library | Future matched comparison after space diagnosis | untested; v8 quarantined | Prospectively frozen primary effect with guardrails and paired uncertainty. |
| C4: graph feedback changes what miners generate | Minimal matched controller experiment | development allowed; claim unearned | Graph beats random and non-graph adaptive controls. |
| C5: the full loop improves factor-library quality | Later matched-budget Quanta backtest | not ready | Cost-aware portfolio improvement survives seed/rolling checks. |
| C6: exported mechanisms are faithful to human understanding | Blind pilot plus S1b interventions | untested | Prospective blind agreement and fidelity rules pass; otherwise downgrade. |

## 6. Comparison and ablation plan

### Baseline layers

1. Reuse the hash-bound Alpha158 + LightGBM anchor when its source/config/data/result identity is verified. Its `best_iter=14` is an early-stopping outcome under the 500/ES50 protocol, not a fixed-14 training regime.
2. Run a typed pure-symbolic library and a type-legal random library under the new project contract because they are direct method controls, not redundant published baselines.
3. For S2, report both replacement and Alpha158-plus-new-library incremental arms where joint support and common library caps are meaningful.
4. Reproduce the expensive Alpha158 anchor only if the precomputed-factor adapter cannot pass a narrow parity smoke against the recorded protocol.

### Core ablations

| Full component | One-change ablation |
|---|---|
| Spline residual | Pure Symbolic-KAN with the same primitive dictionary and budget. |
| Training-time symbolic gates | Free-spline KAN followed by post-hoc symbolification. |
| Residual task routing | Same miners and controller features with the original label. |
| Diversity repulsion | Same graph/task routing with the repulsion coefficient set to zero. |
| Graph message passing | Same-information Flat-Controller. |
| Typed structural evolution | Parameter-only mutation at the same proposal count. |
| Adaptive budget | Uniform budget and Bandit/UCB budget controls. |

## 7. Metrics and fairness

- Primary: net Information Ratio relative to SH000300 under the pinned open-price cost model.
- Secondary: net annualized return, Sharpe, IC, RankIC, ICIR, RankICIR, validation-to-development retention, and incremental value over the selected baseline.
- Guardrails: maximum drawdown, turnover/cost, coverage, effective rank, formula complexity, spline-residual ratio, soft/hard fidelity, decay, and negative-control false-positive behavior.
- Matched resources: data access, seeds, candidate proposals, full evaluations, factor-library cap, joint support, LightGBM, portfolio strategy, transaction costs, and CPU/GPU budget.
- Search size, training rounds, and elapsed time are explanatory process variables, never substitutes for the primary backtest result.

## 8. Honest fallback narratives

| Evidence state | Retained paper story |
|---|---|
| KAN and graph both supported | Full MIRAGE-KAN closed-loop alpha discovery. |
| KAN supported, graph unsupported | Symbolic-Residual KAN and out-of-dictionary primitive recovery. |
| KAN necessity remains unsupported | Preserve the negative result; retain only computational-transparency or benchmark contributions actually supported. |
| Backtest improves, strict fidelity fails | Semi-symbolic predictive factor discovery without mechanism claims. |
| Both method claims fail | Auditable mechanism-recovery benchmark, evaluation contract, and systematic negative result. |

## 9. Figure and writing plan

- Figure 1 teaser: matched-budget cost-aware backtest and library effective-rank comparison, populated only after S4/S5.
- Figure 2 method: standard PIT/evaluator components compressed; Symbolic-Residual miner and the Miner–Factor feedback loop enlarged.
- Main Table 1: matched factor-library backtest.
- Main Table 2: E1–E5 KAN necessity and fidelity.
- Main Table 3: random → non-graph adaptive → graph-controlled generation distribution.
- Main Table 4: component ablations and resource accounting.

Writing proceeds alongside evidence. Once a submission date is frozen: at T−4 weeks freeze story/experiment table and draft Introduction; at T−3 draft Method and figure; at T−2 draft Experiments/Related Work/Abstract; at T−1 run paper-review and visual polish. Evidence gates, not the writing calendar, control method expansion.
