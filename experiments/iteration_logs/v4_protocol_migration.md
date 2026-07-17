# v4 Protocol Migration — Iterative Coder Log

## Iteration 1 (Phase 1/2)

- **Score**: 0.42 (lint=1.0, format=0.0, test=0.0, self=0.55)
- **Lint**: passed (0 issues)
- **Format**: failed (migration test needs Ruff formatting)
- **Tests**: failed (0/3; expected red state)
- **Changes**: `tests/governance/test_v4_protocol_migration.py`
- **Feedback**: tests prove the v4 config and identity do not yet exist; this is the intended pre-implementation failure.
- **Next**: implement the isolated v4 config and protocol constants, then format and rerun.

## Iteration 2 (Phase 1/2)

- **Score**: 0.92 (lint=1.0, format=1.0, test=1.0, self=0.93)
- **Lint**: passed (0 issues in the new protocol/config migration surface)
- **Format**: passed for `protocol.py` and the new migration test; unrelated pre-existing formatting drift remains in three mechanically updated legacy test files.
- **Tests**: passed (6/6 migration assertions, including preserved v2→v3 history)
- **Changes**: `src/mirage_kan/protocol.py`, `configs/experiments/s2a_kan_e3_vertical_v4.yaml`, active v4 test fixtures, `tests/governance/test_v4_protocol_migration.py`
- **Feedback**: v4 differs from v3 only by identity/paths and the two minimum-library-size scalars; every v4 writable path contains a v4 identity and is disjoint from v2/v3.
- **Next**: run the complete focused regression set for every mechanically migrated fixture and inspect residual literals.

## Iteration 3 (Phase 2/2)

- **Score**: 0.95 (lint=1.0, format=1.0, test=1.0, self=0.96)
- **Lint**: passed (0 issues across all files in scope)
- **Format**: passed for the newly authored/changed protocol surface; legacy test-only formatting drift was not broadened into unrelated reformatting.
- **Tests**: passed (87/87 focused regressions)
- **Changes**: no logic changes; verification and residual-literal audit only
- **Feedback**: topology, authority, opening, mining, decision assembly, and five-arm runner fixtures all accept the active v4 identity. Remaining v3 literals are historical migration inputs or belong to the separately owned implementation-lock test.
- **Next**: done; hand off to the parent for base-lock creation and full-suite audit.
