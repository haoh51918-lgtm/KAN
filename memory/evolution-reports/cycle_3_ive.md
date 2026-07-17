# Cycle 3 IVE — S2a v2 pre-label opening failure

> Historical cycle record. The former “new protocol for an implementation bug” rule contributed to the successor loop and is superseded by the red-test/fix/same-path policy in `AGENTS.md`.

## Classification

- **Paper outcome**: `NOT_STARTED (InfrastructureFailureBeforeLabelAccess)`
- **Failure class at the time**: implementation failure then treated as retryable only under a new protocol; this policy is now superseded
- **Research direction**: retained unchanged

## Evidence

The v2 implementation closure passed 304 deterministic tests and was signed.
The first production invocation failed while serializing the mining entitlement:
real YAML dates were `datetime.date` objects, which standard JSON cannot encode.
All seven topology targets terminalized. The entitlement file is an immutable
zero-byte witness because `O_EXCL` occurred before payload encoding.

No label loader, miner, control, factor publication, Quanta adapter, or
development opening executed. This failure contains no evidence for or against
MIRAGE-KAN factor quality.

## Confirmed root cause

Synthetic governance fixtures used date-looking strings and missed YAML's typed
date scalar behavior. The receipt writer also encoded after creating the
exclusive destination, turning a validation error into a consumed path.

## Memory update

Added a reusable rule: governance receipts must canonicalize typed config
scalars and fully encode/validate payload bytes before any no-replace filesystem
claim. Tests must parse the same serialization format used by production.

## Historical actions taken

1. Red-test YAML `date` values in mining and development receipts.
2. Normalize periods to ISO strings at the receipt boundary.
3. Move canonical JSON encoding before `os.open(..., O_EXCL)`.
4. v2 terminal evidence was preserved and a new protocol/topology/lock chain was used; current policy would fix an equivalent reversible bug without reflexively creating a scientific successor.
5. Keep scientific seeds and thresholds unchanged because science never opened.

## Expected impact

The successor changes only receipt typing and publication ordering. It does not
change candidate membership, training, scoring, selection, evaluation, or
decision semantics.
