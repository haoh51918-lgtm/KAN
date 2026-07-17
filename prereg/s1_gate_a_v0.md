# S1 Gate A Preregistration v0

> Frozen before any E1–E5 result is produced. `KAN_Alpha_PR.md` is the method authority. This document fixes the smallest falsifiable test of whether the spline residual has a necessary role; it does not replace the proposal or authorize graph work.

## Question and decision

Can a Symbolic–Residual KAN recover a stable, deliberately out-of-dictionary univariate mechanism from a typed OHLCV-style panel, convert it into a compact executable primitive, and retain the numerical behavior after hardening?

Gate A passes only if the hybrid method supplies a quality–fidelity–complexity advantage that neither the initial symbolic dictionary nor a matched MLP supplies. Prediction accuracy alone is insufficient. A failed Gate A keeps the graph stages locked and activates the KAN-decentered fallback described in the proposal.

## Synthetic data with mechanism truth

Three independent replications use seeds `1729`, `2718`, and `31415`. Each replication has independent train, validation, and test panels with 64 assets and, after a 60-day burn-in, 256, 64, and 128 dates respectively. The generator and all arrays must be saved with a manifest before model training.

For asset \(i\), log close follows a stationary, heteroskedastic return process:

\[
r_{i,t}=0.15r_{i,t-1}+0.012\exp(h_{i,t}/2)\epsilon_{i,t},
\qquad
h_{i,t}=0.90h_{i,t-1}+0.20\eta_{i,t}.
\]

Initialize `Close=100`, `r=0`, `h=0`, and log-volume state `u=0` before burn-in. Then

\[
Open_{i,t}=Close_{i,t-1}\exp(0.003\epsilon^O_{i,t}),
\]

\[
Close_{i,t}=Close_{i,t-1}\exp(r_{i,t}),
\]

\[
High_{i,t}=\max(Open_{i,t},Close_{i,t})\exp(0.006|\epsilon^H_{i,t}|),
\]

\[
Low_{i,t}=\min(Open_{i,t},Close_{i,t})\exp(-0.006|\epsilon^L_{i,t}|),
\]

\[
u_{i,t}=0.75u_{i,t-1}+4|r_{i,t}|+0.25\epsilon^V_{i,t},
\qquad Volume_{i,t}=10^6\exp(u_{i,t}).
\]

Each split uses NumPy `Generator(PCG64(seed + offset))`, with offsets 0, 10,000, and 20,000 for train, validation, and test. For the complete burn-in-plus-split rectangle, standard-normal arrays are drawn in the fixed order `return`, `volatility`, `open`, `high`, `low`, `volume`, and finally `target_noise`; every array has date-major shape `(dates, assets)`. All innovations are independent unless the equations explicitly couple them. The generator version and NumPy version are recorded in each artifact manifest.

The causal driver is

\[
x_{i,t}=\operatorname{clip}\left(\frac{Return(Close,5)_{i,t}}{0.03},-4,4\right).
\]

The clean response uses an asymmetric saturating shape that is absent from the initial primitive dictionary:

\[
g(x)=
\begin{cases}
-0.8(1-\exp(1.5x)), & x<0,\\
 1.4(1-\exp(-0.7x)), & x\ge 0.
\end{cases}
\]

The observed target is `g(x) + noise`, with Gaussian noise standard deviation equal to 10% of the training-panel standard deviation of `g(x)`. Test metrics are reported against both the observed target and the clean truth; clean-truth error is primary for mechanism recovery.

Candidate typed inputs are `Return(Close,w)` for `w ∈ {2,5,10,20}`, `SafeDiv(Sub(High,Low),Close)`, and `SafeDiv(Volume,TsMean(Volume,20))-1`. Correlated return windows are intentional decoys. The exact source/window truth is `Return(Close,5)`. Inputs are standardized using train-only median and interquartile range and applied unchanged to validation and test. Unsupported lookback rows are excluded identically across arms; no imputation is allowed.

## Frozen initial primitive dictionary

The analytical dictionary contains `Identity`, `Abs`, `Square`, `SignedLog1p`, `Tanh`, `Clip(-1,1)`, `PositiveHinge(0)`, and `NegativeHinge(0)`, each with an affine input and output scale where the model class permits it. `Exp`, asymmetric saturation, piecewise exponential functions, and the target formula are forbidden before primitive promotion.

The target family must not appear in initialization, training loss, or E2/E3/E5 candidate generation. It may be consulted only by the evaluator and by the post-discovery audit after the spline shape has been frozen.

## Comparison arms

| ID | Arm | Required implementation and role |
|---|---|---|
| E1 | Free-spline KAN | Actual cubic B-spline KAN without analytical gates; numerical upper bound. |
| E2 | Post-hoc symbolification | Fit E1 first, then independently fit each learned edge to the frozen dictionary using train data only; no joint symbolic retraining. |
| E3 | Pure Symbolic-KAN | Train-time gates over the frozen analytical dictionary; no spline path. |
| E4 | Symbolic + Spline Residual KAN | Same analytical path as E3 plus a penalized cubic B-spline residual and explicit residual-energy accounting. |
| E5 | Typed symbolic regression | Search executable typed ASTs using only the frozen dictionary and candidate inputs; no spline or neural proxy. |
| C6 | Matched MLP | SiLU MLP matched to E4 within 10% trainable parameters; numerical non-KAN control, never presented as a formula. |

No arm may be implemented by relabeling another estimator, fitting the known target formula, or routing calculations through a NumPy/Pandas proxy when the arm requires Torch optimization.

## Training, selection, and budget

- Neural arms use float64 for the primary run, AdamW, learning rate `0.003`, batch size `2048`, at most `2000` optimizer steps, and early stopping patience `200` validations. Validation occurs every 20 steps.
- Neural arms may expose at most `10,000` trainable parameters. E3 and E4 share the same graph, width, input gates, optimizer, and step ceiling; their only structural difference is the residual spline path. C6 is parameter-matched to E4 within 10%.
- E1 and E2 share one fitted E1 model. E2 receives at most `8 × number_of_fitted_edges` primitive fits and cannot retrain the spline model.
- E5 receives at most `12,000` distinct valid AST evaluations, depth at most 5, and node count at most 15. Duplicate canonical ASTs consume an attempted-candidate ledger entry but not another full fit.
- Each arm has one frozen default configuration. Up to two implementation-recovery attempts are allowed only when a reference sanity check fails; they are logged and cannot be selected by test performance.
- Model selection uses validation clean-target NRMSE followed, within `0.005` NRMSE, by lower complexity. Test is read once after the selected checkpoint/configuration is frozen.
- Wall time, sample presentations, parameter count, valid candidates, and estimated FLOPs are reported. They explain efficiency but do not override the scientific gate.

## Hardening and primitive governance

E4 first trains with the target family hidden. Its residual edge is sampled on the fixed grid `[-4,4]` with 801 equally spaced points. A discovery is eligible only when the same input/window is selected and the aligned residual shapes have pairwise Pearson correlation at least `0.95` in at least two seeds.

Only after eligibility is established, fit a low-complexity monotone piecewise family to the frozen residual samples using train/validation samples. Candidate families and their description lengths must be declared in the run manifest before their fits are read. The test clean target cannot select the family. The promoted primitive receives a new versioned ID and must pass the proposal’s domain, boundary, stability, non-duplication, and executable-implementation checks.

Hardening replaces the E4 residual by that promoted primitive, freezes the input/window, and refits only continuous affine scales on train plus validation. If no governed primitive passes, the result remains semi-symbolic and Gate A fails.

## Metrics

For every seed and arm report:

1. clean and noisy test NRMSE (`RMSE / std(test clean truth)`);
2. shape NRMSE on the fixed 801-point grid;
3. exact source/window recovery and selected-input mass;
4. soft-to-hard Pearson correlation, NRMSE, and maximum absolute error;
5. residual spline energy ratio;
6. executable AST node count, depth, number of free constants, and serialized description length;
7. trainable parameters, steps, candidate evaluations, FLOPs estimate, wall time, and peak memory.

Aggregate by median and full seed range. Add a date-block bootstrap confidence interval within each seed, but do not claim a population p-value from three seeds.

## Negative controls

For E4, rerun one matched budget per seed with the observed labels permuted by date blocks of length 20. A healthy pipeline must not promote a primitive on any of the three null runs. The generator also verifies that removing `Return(Close,5)` makes the specified mechanism unrecoverable rather than silently reconstructing it through future access.

## Gate A thresholds

Gate A passes only when all conditions hold:

1. **Capacity sanity:** E1 or C6 achieves median clean test NRMSE at most `0.15`; otherwise the run is technically inconclusive, not evidence against KAN.
2. **Stable recovery:** E4 selects `Return(Close,5)` and produces an eligible residual shape in at least two of three seeds.
3. **Shape quality:** E4 median pre-promotion shape NRMSE is at most `0.12`, every successful seed is at most `0.18`, and the median is at least 15% lower than the better of E3 and E5.
4. **Numerical competitiveness:** E4 median clean test NRMSE is no more than 15% above the better numerical upper bound from E1 and C6.
5. **Executable promotion:** the hardened promoted-primitive model has median clean test NRMSE at most `0.15`, soft–hard correlation at least `0.98` in every successful seed, and median soft–hard NRMSE at most `0.10`.
6. **Interpretability advantage:** E4’s hardened result is Pareto-undominated across clean NRMSE, hardening fidelity, and description length; neither E3/E5 nor E2 may be no worse on all three axes and strictly better on one.
7. **Null safety:** zero of three permuted-label runs promotes a primitive.

Gate A fails scientifically when the capacity sanity passes but any of conditions 2–7 fails after valid runs and the allowed implementation-recovery attempts. It is invalidated and rerun when parity, causal masking, split isolation, budget accounting, or test-once rules fail.

## Evidence boundary

Passing this synthetic gate shows only that the KAN residual can discover and harden one controlled out-of-dictionary mechanism. It does not show that MIRAGE-KAN finds profitable Alpha, that graph feedback helps, or that the promoted primitive transfers to markets. Those claims remain owned by S2 and later gates.
