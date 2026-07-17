# Decision: Seal Gate A and activate staged Plan C

- **Date**: 2026-07-16 UTC
- **Authority**: `KAN_Alpha_PR.md`, especially Sections 21.2, 21.4, and 22.4
- **Evidence**: `artifacts/s1_gate_a_scientific/s1_gate_a_v1_scientific_formal_v1/`
- **Decision status**: partially superseded (2026-07-17) — see supersession note at end

## Decision

Seal S1 Gate A v1 as a valid `scientific_fail`. KAN is no longer a required backbone for the main research claim. It becomes one optional miner in a heterogeneous factor-generation system.

Activate Plan C in stages:

1. Build and evaluate the real S2 factor library before implementing graph control.
2. Use typed GP / symbolic miners as the executable primary path; retain MLP and semi-symbolic KAN only as explicitly labeled auxiliary miners.
3. Keep every miner on the same S0 factor-library publication and Quanta evaluation path.
4. Unlock graph work only after S2 establishes portfolio value and S3 establishes a material redundancy or allocation problem that simple heterogeneity does not remove.
5. Gate B must compare graph control against Random, Independent, Boost-Sequential, Bandit-Budget, and same-information Flat-Controller at matched budgets.

## Trigger evidence

- Gate 1/2/3/4/7 passed.
- Gate 5/6 failed because no governed HARD primitive was produced.
- E4 was numerically strong but semi-symbolic: clean NRMSE median `0.0233467`, shape NRMSE median `0.0159897`.
- Exact-family sanity passed, while all three frozen residual-family fits failed consistently across seeds.
- Independent diagnosis classified the failure as architectural decomposition non-identifiability, not an implementation defect.

## Rejected actions

- Do not lower governance thresholds after observing the result.
- Do not relabel the nonmonotone residual correction as the hidden primitive.
- Do not reuse the opened v1 test splits for a recovery run.
- Do not claim Gate A success from E4 prediction quality alone.
- Do not let an optional S1b delay S2's real factor-library and Quanta backtest.

## Optional non-blocking branch

A future S1b may prospectively study promotion of the centered full selected edge or hard-first / sequential decomposition. It is a new method and needs a new preregistration, new scientific seeds, its own one-shot test opening, and a distinct claim. It is not an implementation recovery of v1.

## Supersession note (2026-07-17, append-only)

The 2026-07-17 principal directive (`KAN_Alpha_PR.md` Section 25;
`governance/decisions/2026-07-17_kan_interpretability_first_directive.md`)
abandons this decision's KAN-de-centering path outright: KAN is the
non-negotiable mining core, MLP is excluded from the miner pool
(matched-capacity control only), typed GP/SR is demoted to a comparison
baseline arm, and proposal Section 21.4 now carries a deprecation notice
forbidding future attempts at this path absent a new written principal
decision. The standing mainline is MIRAGE-KAN unless the principal actively
changes course.

Items of this decision that survive unchanged: the Gate A v1 seal and seed
retirement (trigger evidence), real-data S2 priority before graph work
(item 1), the shared S0 publication/evaluation path (item 3), graph-unlock
staging (item 4), and the Gate B baseline discipline (item 5). Item 2's miner
hierarchy is void; S1b is upgraded from optional to a mandatory mainline
experiment (still non-blocking for S2). The original text above is preserved
verbatim.

