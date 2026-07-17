# S2a v8 authority-receipt canonical-hash corrective successor

- **Date**: 2026-07-17 UTC
- **Predecessor**: `s2a_kan_e3_vertical_v7`
- **Successor**: `s2a_kan_e3_vertical_v8`
- **Failure phase**: after v7 implementation lock, before mining-rebind receipt
- **Development opening consumed**: no
- **Label access**: no

## Failure

The first real v7 rebind-receipt build failed before its exclusive output path
was created. The verifier recomputed the source authority receipt's internal
`receipt_sha256` over compact sorted JSON. The authority subsystem defines that
identity over indented, sorted JSON followed by one newline. The synthetic test
fixture shared the compact-JSON assumption, so it did not expose the mismatch.

No source artifact was changed, no receipt was partially written, no mining or
development entitlement was issued for v7, and no label was loaded.

## Correction

v8 changes only the authority-receipt canonical-byte reconstruction in the
cross-protocol rebind verifier and its fixture. It uses the same canonical form
as `governance/authority.py`: `json.dumps(..., indent=2, sort_keys=True)` plus a
trailing newline. A regression must verify the real v6 sequence-1 receipt SHA.

All v7 scientific fields, the exact v6 whole-topology source, the sparse-PIT
checkpoint projection, graph authorization, evaluation criteria, and custody
constraints remain unchanged. v7 locks are preserved as terminal pre-opening
software evidence and are never rewritten.
