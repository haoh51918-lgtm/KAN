# S2a KAN-centered complete-chain preregistration v2

## Status and authority

This is a prospective development-screen protocol under the sole authority of
`KAN_Alpha_PR.md`, especially Section 25. It replaces the superseded Plan C v1
scientific design; it does not alter or reopen any sealed v1 artifact. WIKI,
predecessor reports, and v1 candidate membership are not scientific authority
and cannot supply v2 candidates or thresholds.

The purpose of S2a v2 is to run the smallest faithful complete chain from a real
pure-symbolic KAN miner to an immutable executable factor library and the pinned
Quanta backtest. It may authorize formal S2, but cannot establish a paper claim,
formally release an interpretable library, or unlock graph work.

## Research question

Can a non-graph population of pure-symbolic E3 KAN miners produce a fully
traceable strict-formula factor library that is non-inferior to Alpha158 and a
matched-capacity black-box control under the frozen Quanta protocol, while not
being jointly Pareto-dominated by a matched-budget typed GP/SR control?

Outperformance is secondary strengthening evidence. Interpretability cannot be
used to excuse statistically material underperformance.

## Production miner definition

The production miner is a genuine differentiable E3 KAN search process, not a
label attached to a formula discovered elsewhere:

1. its only external data inputs are PIT raw Open, High, Low, Close, and Volume;
2. a frozen internal typed layer evaluates legal temporal atoms;
3. two categorical KAN edges learn differentiable gates over those atoms;
4. temperature annealing and a final straight-through hard-forward stage train
   the gates against mean daily cross-sectional Pearson IC;
5. the hardener receives checkpoint logits, atom manifest, and frozen
   temperature only, and emits exactly one `Sub(atom_positive, atom_negative)`
   DSL formula per miner; and
6. the final factor is recomputed independently by the common DSL executor.

The model has no B-spline path, so every accepted factor is level 1 with exact
`spline_ratio=0`. A formula is `kan_mined=true` only when its ledger contains the
complete KAN seed, initial/final gates, temperature/optimization trace, gradient
receipt, hardening selection, rejected alternatives, and independent AST replay.

It is forbidden to seed the miner with a complete GP formula, let KAN merely fit
coefficients on a GP-selected structure, or pass a GP formula through KAN and
relabel it. Final representability by a KAN is not evidence of KAN provenance.

## Search topology and budget

Four heterogeneous profiles use different legal typed atom banks: short price,
long price, reversal, and price-volume. Each profile has 64 independently seeded
KAN miners, for exactly 256 budget-consuming attempts. Each miner hardens once
and contributes at most one candidate. No hidden retry or top-k expansion is
allowed. Failed, duplicate, low-fidelity, unsupported, and rejected candidates
all consume budget and enter the ledger.

Every miner uses 300 Adam updates, float64, temperature 2.0→0.10, and a final 25%
hard straight-through phase. A deterministic 80% moving-block bootstrap of
training dates, with 20-day blocks, creates miner diversity without reading
validation outcomes. Seeds and their derivation are frozen in the YAML config.
Within each profile, atoms are deduplicated and ordered by canonical hash. Both
categorical edges use that same ordered bank, so index order can never become an
implicit scientific choice.

Atom families lower exactly as follows: return is `Return(price,w)`;
price-vs-mean and mean-vs-price are the corresponding ordered difference divided
by the denominator shown in the YAML; lag-vs-price is
`(Delay(price,w)-price)/Delay(price,w)`; volume change is
`Delta(Volume,w)/Delay(Volume,w)`; and volume-vs-mean is
`(Volume-TsMean(Volume,w))/TsMean(Volume,w)`. Price families enumerate only
Open/High/Low/Close. Volume families always use Volume and ignore a profile's
price-field list.

For each miner, NumPy PCG64 draws 20-day training blocks uniformly with
replacement from all non-wrapping legal block starts. Blocks are concatenated
in within-block chronological order and truncated to `ceil(0.8 × train dates)`.
Repeated dates remain as multiplicity weights in the objective; the last block
is truncated and wraparound is forbidden.

The library cap is 16 and minimum size is eight. At least three profiles must be
represented. Every admitted factor needs coverage at least 0.85, absolute train
and validation RankIC at least 0.005, matching signs, soft-hard Pearson at least
0.98, soft-hard NRMSE at most 0.10, gate probability margin at least 0.05, and
absolute validation Spearman below 0.80 against already-selected factors.
Padding is forbidden.

## Data access

- Discovery train: 2016-01-01 through 2020-12-31.
- Candidate validation: 2021-01-01 through 2021-12-31.
- Development portfolio test: 2022-01-01 through 2025-12-26.
- Label: verified one-day `fwd`, identical to Quanta's
  `Ref($close,-2)/Ref($close,-1)-1`.

Because this label uses the next two trading dates, the final two trading dates
of both the train split and the validation split are purged from all objectives,
admission statistics, and controls. A label whose price horizon crosses a split
boundary is forbidden even when the cached value is finite. Rolling features
retain exactly the preceding 60 raw-only trading dates as warm-up; a cache with
fewer than 60 pre-train dates fails closed. Warm-up labels are null and cannot
enter an objective. Thus the same
frozen AST has identical in-split values during mining and later full-panel
replay, without exposing a label or price horizon from the next split.

Mining artifacts contain formula/parameter lineage and replay evidence only
through the validation end. They must not read or publish 2022--2025 raw values.
After the single development opening is consumed, the evaluation runner
independently replays the frozen ASTs and MLP final parameters on the full raw
PIT panel. Development predictions stored before that opening are forbidden.

Train labels drive KAN/GP/MLP fitting. Validation labels drive only frozen
admission, fidelity, and method-screen metrics. Development outcomes are opened
once, after the production library and all control artifacts are immutable.

The live proposal hash is revalidated before first label access, before each
arm, before each publication, before development opening, and before decision
publication. Any drift terminalizes the run as `superseded_authority`.

## Five fixed evaluation arms

| Arm | Role | May enter production library? |
|---|---|:---:|
| `alpha158_replay` | infrastructure and baseline anchor | no |
| `kan_e3_selected` | sole production candidate library | yes, subject to release gates |
| `typed_gp_sr_control` | same-DSL, same-budget discrete-search method control | no |
| `matched_blackbox_control` | capacity-matched MLP falsification control | no |
| `kan_e3_permutation_control` | complete KAN rerun on within-date permuted labels | no |

GP/SR uses a new seed and a new 256-attempt topology. It shares the hard formula
space, executor, split, admission, full-evaluation budget, and final library size,
but its structure search is discrete and `kan_mined=false`.

Its matched formula space is exactly `Sub(atom_positive, atom_negative)` from
the same profile bank used by KAN. Per profile, 16 initial pairs are drawn
uniformly; a same-atom draw is an invalid attempt and is not retried. Each of the
remaining 48 attempts chooses mutation or crossover with equal probability.
Parents are selected by a four-way tournament with replacement over prior valid
individuals, ordered by absolute train RankIC, AST size, then canonical hash.
Mutation uniformly replaces one edge with a different bank atom. Crossover takes
the positive edge from the first tournament parent and negative edge from the
second. Cancellation and duplicates consume budget. Validation is inaccessible
during evolution and is used only by the common final admission rule.
If no valid prior parent exists, the attempt is recorded as `invalid_no_parent`
without a retry.

The MLP uses the paired profile's internal atom values only as an evaluation
instrument, a two-unit SiLU hidden layer, and the same updates and bootstrap
budget. Its trainable-parameter gap from the paired KAN must be at most 10%.
Its outputs are published only as a non-promotable control panel, never a factor
library.
One MLP is fitted for every selected KAN factor using that factor's exact profile
and block-bootstrap weights; its seed is the frozen MLP base plus the KAN global
attempt index. This fixes the black-box output count and prevents post-hoc choice
of an easier bootstrap.

The permutation arm produces two distinct records. The false-positive ledger
applies the real production efficacy thresholds and must contain no admitted
candidate. Separately, a size-matched null library is selected from structurally
valid, coverage-valid candidates by permuted train/validation score with the
same profile, diversity, and tie-breaking rules, but without requiring destroyed
labels to exceed the real efficacy threshold or agree in sign. This null library
exists only for a fair Quanta combination comparison.

## Interpretability and release boundary

Before development opening, 100% of production candidates must have canonical
ASTs, independent replay, complete lineage/hardening receipts, and mechanism
cards. Cards include variables, windows, complexity, gate/shape summaries,
soft-hard fidelity, variable and lag interventions, local counterfactual response,
one-sentence mechanism, applicability, and failure conditions.

All intervention evidence is computed on the frozen 2021 validation panel, not
on development outcomes. For each raw variable used by the selected AST, the
variable-contribution ablation replaces every selected edge contribution using
that variable with zero while holding the other selected edge fixed. Lag-band
ablations apply the same operation to selected edges whose window lies in the
inclusive short (2--5), medium (10--20), or long (40--60) band. Each ablation
reports mean absolute factor change divided by the baseline factor's population
standard deviation and baseline-versus-ablated Pearson correlation on the
point-in-time rows where both outputs are finite. A zero or non-finite baseline
standard deviation is a mechanism-evidence failure. These are explicitly edge
contribution ablations, not claims that a NaN-filled raw market panel is a valid
economic counterfactual.

For each of the two selected atoms, the local counterfactual moves that atom's
output by plus and minus one validation-population standard deviation while
holding the other edge fixed. It records the mean signed factor change and its
direction after applying the terminal `Sub` sign. A zero or non-finite atom
standard deviation fails the mechanism evidence. Primitive shape plots are
exact analytic DSL responses on the inclusive input-ratio grid 0.50--1.50 in
steps of 0.01; they report primitive output, signed edge contribution,
monotonic direction, and saturation intervals.

The anonymized blind-review package hides method and PnL. Formal use of the word
"interpretable" requires at least two human quantitative reviewers to restate
the mechanism and predict perturbation directions at the frozen accuracy gate.
Autonomous agents may QA the package but cannot be recorded as human reviewers.
The held-out questions ask for the direction of the frozen plus-one-standard-
deviation local perturbation on each selected edge. Their answer key is the
mechanism card's local-counterfactual record, is excluded from the blind package,
and both questions and answer key are immutably published before human review.
Until genuine review is complete, the library is labeled
`computationally_transparent_not_yet_interpretable`. This pending human gate does
not change membership and therefore does not block the developmental backtest.

## Quanta and statistical decision

All five arms use the pinned real Quanta path, LightGBM 500 rounds with early
stopping 50, TopkDropout 50/5, open-price execution, and the frozen cost model.
The Alpha158 replay is infrastructure-valid only within IR/MDD/RankIC absolute
tolerances 0.03/0.02/0.003 from the historical anchor.

The primary performance statistic is a paired 20-trading-day moving-block
bootstrap with 10,000 replicates over daily net benchmark-relative return. The
one-sided 95% lower bound of `IR(KAN) − IR(Alpha158)` must be at least -0.022,
which is the prospectively frozen 10% of the absolute 0.22 anchor. The analogous
KAN-minus-MLP lower bound must be at least `-0.1 × |MLP IR|`.
Each replicate draws non-wrapping 20-day block starts uniformly with replacement,
concatenates and truncates to the locked calendar length, and applies identical
indices to both arms. IR is `sqrt(252) × mean / sample_std(ddof=1)`; the lower
bound is the empirical 0.05 quantile of paired delta IR.
Any zero or non-finite resampled variance is an infrastructure failure; invalid
replicates may not be silently dropped or replaced.

Guardrails are KAN-minus-Alpha158 RankIC at least -0.003, MDD at least -0.03,
mean turnover and realized-cost ratios at most 1.20, and at least three of four
calendar years with KAN active return not below Alpha158. Integrity additionally
requires 8–16 strict factors, at least three profiles, cards for every factor,
zero permutation false positives, and a valid replay.

The GP/SR arm must not Pareto-dominate KAN simultaneously on development delta-IR
lower bound, unique admitted factors per full evaluation, and selected-library
effective rank. A GP or MLP advantage can only shrink the KAN claim; it can never
promote either control into the production miner pool.

Effective rank is computed on 2021 validation rows where every selected factor
is finite and in-universe. Each factor is population-z-scored cross-sectionally
per date; the pooled row correlation eigenvalues are floored at zero and
normalized to `p`, then effective rank is `exp(-sum(p log p))`. GP Pareto
dominance means greater-or-equal on all three endpoints and strictly greater on
at least one. Both delta-IR lower bounds are method-minus-Alpha158; admitted rate
uses an exact eligible-count numerator over 256. Floating comparisons use the
YAML-frozen `1e-12` tolerance.

If every machine gate passes but human blind review is pending, the outcome is
`advance_s2_formal_pending_human_blind_review`. S2a always records
`formal_promotion_allowed=false`; formal S2 requires a new lock and cannot reuse
this development period as final evidence.

## Required artifacts

- proposal/config/preregistration/implementation lock chain;
- append-only KAN, GP/SR, permutation, MLP, rejection, and data-access ledgers;
- every KAN checkpoint gate trajectory and hardening receipt;
- immutable production and control libraries/panels with exact provenance;
- one mechanism card per production factor and an anonymized blind package;
- one current Alpha158 replay and four real custom Quanta evaluations;
- exact daily return, turnover, cost, prediction coverage, bootstrap output,
  machine decision, and a human-readable Chinese comprehensive report.
