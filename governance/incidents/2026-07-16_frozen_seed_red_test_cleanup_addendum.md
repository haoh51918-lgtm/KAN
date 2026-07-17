# Automatic cleanup addendum for the frozen-seed red-test incident

Relation: this is an append-only custody update to
`2026-07-16_frozen_seed_red_test_incident.md`. It does not replace or alter the
original incident snapshot or adjudication.

## Discovery

- Absence discovered: `2026-07-16T22:24:01.306159Z`.
- Missing historical root: `/tmp/pytest-of-root/pytest-134/test_matrix_modes_reject_scien0/s1_gate_a_scientific/fixed_project_artifacts`.
- At discovery, pytest retained only session roots `pytest-136`, `pytest-137`,
  `pytest-138`, and the `pytest-current` link. The parent directory mtime was
  `2026-07-16T22:18:40.457338677Z`.

## Custody finding

The historical root was automatically removed by pytest's temporary-directory
retention cleanup while later test sessions were created. No project command or
manual action deleted the root. The cleanup was discovered during the final
fresh-smoke integrity audit; it was not requested or relied on by the research
workflow.

The original incident report's eight-file path, byte-size, mtime, and SHA-256
inventory remains the historical evidence captured while the tree existed. The
temporary files are no longer present and therefore cannot be re-hashed. They
must not be reconstructed or regenerated: doing so would fabricate custody and
could trigger another frozen-seed execution.

## Governance effect

The cleanup changes artifact availability, not the adjudication. The partial
attempt remains classified as `pre-test partial scientific attempt,
invalidated`. No test claim, opening, prediction, receipt, test metric, Gate
report, or success manifest was produced. Test-once entitlement remains
unconsumed, and the run remains eligible for one clean execution only after the
independent audit passes.
