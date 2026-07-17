# Incident: S2a v2 mining-entitlement serialization failure

- **Date**: 2026-07-17 UTC
- **Classification**: `pre_label_infrastructure_failure`
- **Run disposition**: `terminal_non_scientific_run`
- **Severity**: execution high; scientific contamination none
- **Status**: contained; successor protocol required

## What happened

The locked `s2a_kan_e3_vertical_v2` mining entry point successfully verified the
implementation closure, preclaimed and claimed all seven mining artifact
targets, and revalidated proposal authority. It then attempted to write the
single-use mining entitlement. The live YAML parser represented the unquoted
train and validation dates as `datetime.date` objects, while the opening writer
passed those objects directly to `json.dumps`. JSON serialization failed with
`TypeError: Object of type date is not JSON serializable`.

The exclusive writer had already created the entitlement path before encoding
the payload, so the path remains as an immutable zero-byte failure witness.
The mining orchestrator terminalized all seven claimed targets. They must not be
deleted, resumed, or relabeled as publishable artifacts.

## Scientific custody

1. No PIT labels were loaded. The failure occurred inside
   `consume_mining_entitlement`, before `_load_locked_mining_inputs`.
2. No KAN, permutation KAN, GP/SR, or MLP training began; both GPUs remained
   idle apart from runtime initialization.
3. No factor candidate, factor library, mechanism card, Quanta evaluation, or
   2022--2025 development result was produced or observed.
4. The v2 implementation lock and all terminal targets remain immutable.
5. v2 cannot be retried. A successor must use a new protocol ID, base lock,
   implementation lock, opening paths, artifact paths, and topology.

## Root cause

Tests constructed protocol dates as strings and therefore did not exercise the
real YAML scalar type. The opening receipt lacked a canonical scalar-normalizing
boundary before exclusive JSON publication. The same latent defect also
affected `development_period` in the later development opening.

## Required successor correction

1. Add red tests using actual YAML date scalars for both mining and development
   opening receipts.
2. Canonicalize every period value to ISO date strings before opening payload
   construction, and serialize the complete payload before `O_EXCL` creates the
   destination.
3. Keep all scientific design, thresholds, budgets, and seeds unchanged because
   no label or candidate computation was opened.
4. Re-run the complete deterministic test suite, issue a new implementation
   lock, and only then authorize a new mining topology.

