# S1b decomposition-identifiability repair preregistration v1

## Purpose and claim boundary

S1b answers a new prospective question required by proposal Sections 25.5–25.7:
can shape-constrained residuals (R1), sequential hard-first decomposition (R2),
or shape-level discrete decomposition (R4) repair the non-identifiability exposed
by sealed Gate A v1? It does not rerun or reinterpret Gate A v1. Seeds
1729/2718/31415, all v1 checkpoints, and all v1 test outcomes remain retired.

S1b runs in parallel with S2 and never blocks the real-data complete chain. Its
winning arm may become MIRAGE-KAN's second miner only after this protocol passes.

Two conclusions are kept separate. `mechanism_recovery_success` means a method
recovers variables, windows, behavior, and a hard formula. `repair_superiority`
requires improvement over a new-seed E4-joint control; recovery alone cannot be
presented as proof that a repair solved the v1 failure.

## Evidence layers and battery

The experiment has nine mechanism classes and ten task instances. M8 contains
both a linear return and a strictly monotone Tanh transform and is evaluated as
one ranking-equivalence class. The other mechanisms cover 5-day reversal,
20-day trend, volume-confirmation hump, volatility threshold, asymmetric
downside, regime switching, a matched noise distractor, and a new
out-of-dictionary asymmetric saturation.

M9 is deliberately different from v1. For standardized input `x`, its negative
branch is `-0.75(1-exp(1.7x))` and its nonnegative branch is
`1.40x/(1+0.55x)`. The exact piecewise truth is forbidden from the initial
dictionary and R4 family set.

The synthetic layer uses 64 assets and fixed 256/64/128 train/validation/test
dates after 60 burn-in dates. The semi-real layer uses PIT CSI300 raw OHLCV only
through 2021. It circularly block-shifts real 5-day future returns by at least 60
days to preserve nuisance tails, volatility clustering, and missingness, then
injects the known mechanism at train-calibrated SNR 1. Validation/test never
refit scalers or injection magnitude. Recovery is scored against the clean
injected component; the observed mixture is training input only.

## Arms and exact matrix

Three primary arms are E4-mono/R1, E4-seq/R2, and E4-shapeSR/R4. E4-orth/R3 is
a complete synthetic-battery ablation. New-seed E4-joint is the attribution
control; E3 Symbolic-KAN, E5 typed SR, and capacity-matched C6 MLP are
falsification controls. E1/E2 are not rerun.

The full synthetic and semi-real matrices each contain seven arms × ten tasks ×
three seeds = 210 cells. R3 adds 30 synthetic cells, for 450 scientific cells.
R3 cannot enter winner selection. Every paired cell has the same 2,000-step,
10,000-parameter, or E5 12,000-AST ceiling.

R1 enforces monotonicity in the forward parameterization, exact zero anchoring,
and saturation; post-hoc isotonic projection is forbidden. R2 must publish
bitwise freeze receipts for its analytical-only, residual-only, alternating, and
fixed-budget joint-finetune phases. R3 projects on the full training empirical
measure against both selected analytical outputs and the four-affine closure;
minibatch-only correlation penalties are invalid. R4 first learns total shape,
then selects from one globally frozen, mechanism-blind family set by prediction
error plus MDL; task identity, truth, and test outcomes are inaccessible.

## Recovery, fidelity, and negative controls

A successful cell recovers the correct variable/window and full interaction
active set, achieves clean NRMSE ≤0.15, and leaves no unnamed free spline after
hardening. One-dimensional shape thresholds, soft-hard Pearson ≥0.98, median
soft-hard NRMSE ≤0.10, and residual cross-seed Pearson ≥0.95 are inherited only
where their metric semantics match Gate A. Two-dimensional M3/M6 intervention
surfaces use the newly frozen NRMSE ≤0.18 plus all registered response directions.
M7 noise contribution energy must be ≤5%.

Train and validation receive separate 20-date block permutations. Test is never
used for null selection. M9 additionally receives within-instrument, within-date,
sign-flip, and mechanism-free controls. Any null promotion fails an arm.

A mechanism class is recovered with at least two of three seeds in each layer;
M8 requires both instances in the same seed. An arm passes with at least 7/9
synthetic and 6/9 semi-real classes, M9 success in at least two seeds per layer,
R5 reuse compression, zero null promotions, and complete custody.

## R5 cross-task reuse compression

The M9 candidate primitive freezes its internal shape on its origin task. Six
reuse tasks may refit only input/output affine parameters. The primitive's own
description length is charged once and amortized over the corpus. Passing needs
improvement on at least five tasks, corpus MDL reduction ≥10%, median validation
AST evaluations reduction ≥20%, spline-fallback absolute reduction ≥1/3, and
clean validation NRMSE worsening ≤0.01. Numerical equivalence at NRMSE ≤1e-8 is
rejected. The old 0.05/0.995 shape thresholds remain diagnostics, not promotion
authority.

## Atomic test opening and decision

All 450 cells share one atomic test opening. Before it exists, every training
run, validation choice, hard program, M9 primitive, R5 decision, attempt ledger,
and source/data/config identity must be frozen. R1 cannot be viewed before
deciding whether to open R2/R4. There is no recovery or selective rerun after
opening.

S1b passes if any primary arm passes. Repair superiority additionally requires
at least +2 of the six M9 seed-layer cells versus E4-joint without decreasing
the macro mechanism count. Winner ties use common recovered classes, M9 cells,
hard clean NRMSE, amortized MDL, then fixed R1→R2→R4 order.

The 7/9, 6/9, M7 5%, surface 0.18, R5 compression, and semi-real SNR thresholds
are explicitly new normative S1b choices, not disguised Gate A evidence. They
may not be recalibrated using reduced scientific batteries or validation reads.

