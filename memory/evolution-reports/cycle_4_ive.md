# Cycle 4 IVE — S2a v3 strict-library admission failure

> Historical cycle record. The adaptive-successor decision was consumed by later v4-v8 history and is not a current authorization.

## Classification

- **Paper outcome**: `NOT_STARTED (ScientificScreenFailureBeforeDevelopment)`
- **Failure class**: method-yield / count-gate mismatch
- **Historical research decision**: MIRAGE-KAN was retained and an adaptive successor was created; that authorization is exhausted

## Evidence

The v3 run consumed a valid real-label entitlement and completed the production
KAN, permutation KAN, and typed GP/SR search/scoring computation. Strict KAN
selection returned seven factors with the profile quota satisfied, but the
frozen minimum library size was eight. The run failed closed before MLP control,
artifact publication, development opening, and Quanta evaluation.

No development-period result was observed. The failure says nothing about the
portfolio quality of the seven-factor library because the count gate prevented
the primary experiment from running.

## Root cause

The protocol treated an eight-factor minimum as an integrity prerequisite even
though the project's primary horizontal metric is the cost-aware Quanta
backtest of whatever strict library is mined. This allowed a secondary count
metric to block the complete chain by one factor.

The atomic all-or-nothing mining bundle also preserved no candidate-level
failure diagnostic. Custody is sound, but future terminal receipts should carry
aggregate rejection counts so failed attempts remain scientifically useful
without publishing a partial library.

## Historical evolution decision

This is not grounds to lower per-factor quality thresholds or hand-author a
factor. The successor changes only the developmental library-count floor from
eight to six and mirrors that value in the integrity gate. Six is chosen as a
prospective exploratory floor with two factors per required profile on average;
it is deliberately not fitted to the observed count seven.

All efficacy, fidelity, diversity, profile, control, backtest, and decision
rules remain unchanged. Formal promotion remains forbidden.

## Additional infrastructure observation

Independent pre-development audit found that v3's implementation lock omitted
the actual Quanta runtime dependencies `pyqlib` and `lightgbm`. Development was
not opened. The successor must use an isolated, fully inventory-locked runtime
covering Torch, QLib, LightGBM, MLflow, every transitive distribution, wheel
hashes, Python, CUDA/cuDNN, Quanta, and provider identities.

## Reusable memory

1. Do not let an arbitrary library-count floor block the first end-to-end
   backtest when every retained factor meets the real quality rules.
2. Keep count adaptations separate from efficacy-threshold adaptations.
3. On atomic selection failure, persist aggregate rejection diagnostics inside
   the terminal receipt without publishing partial scientific artifacts.
4. A backtest implementation lock must enumerate the packages imported only at
   evaluation time, not merely the packages imported while the lock is built.
