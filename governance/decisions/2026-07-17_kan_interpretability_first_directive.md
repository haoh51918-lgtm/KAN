# Decision: Principal directive — KAN interpretability-first mainline

- **Date**: 2026-07-17 UTC
- **Authority**: Direct principal (research owner) directive, written into
  `KAN_Alpha_PR.md` Section 25 per the owner's instruction. Section 25 is now
  part of the sole proposal authority and supersedes conflicting older
  sections (adjudication table in 25.9).
- **Trigger**: Not an experimental result. This is a research-goal
  clarification issued after reviewing the sealed Gate A v1 `scientific_fail`
  and an external-literature feasibility survey of decomposition
  identifiability repairs.
- **Decision status**: partially superseded; retained as dated research-direction history

> **Current role (2026-07-17):** the KAN/interpretability research motivation is
> retained in the Living Manual. This document's claims that the proposal is the
> sole authority and that graph work remains globally locked are superseded by
> `AGENTS.md`, the Living Manual, and the later graph-unlock decision. It does not
> authorize a new experiment or override a frozen protocol.

> **Subsequent factual correction (2026-07-17):** the proposal was revised
> while the already-entitled v1 mining process was still running. That process
> had read 2016–2021 discovery/validation labels but never opened a 2022–2025
> outcome or Quanta result. Its terminal output is classified as
> `superseded_non_scientific_run`. See
> `governance/incidents/2026-07-17_s2a_v1_inflight_authority_supersession.md`.

## Directive summary

1. The core motivation of the program is KAN's high interpretability as the
   primary claim. Performance parity (non-inferiority) with the baseline is
   acceptable; outperformance is secondary evidence, not a precondition.
2. Mined factors must be mechanistically explicit — human-readable,
   analyzable, and traceable ("where did this factor come from") — not
   black-box search hits. Operational definition and blind-review acceptance
   gate: proposal Section 25.2.
3. KAN is the non-negotiable core of the mining production path. MLP and
   other non-hardenable black-box structures are excluded from the miner
   pool and from all factor production. They remain only as matched-capacity
   falsification controls inside evaluation protocols; control outputs never
   enter the factor library.
4. Typed GP/SR is demoted from "primary executable path" (Plan C decision
   item 2) to a comparison-baseline arm.
5. S1b (decomposition-identifiability repair study) is upgraded from an
   optional branch to a mandatory mainline experiment (three arms:
   shape-constrained residual / sequential hard-first / shape-level discrete
   decomposition; mechanism battery per proposal §15.8; new preregistration,
   new seeds, one-shot opening). It still must not block S2's real-data
   chain.
6. S2's miner pool is re-centered on KAN: pure-symbolic KAN (E3
   architecture) as the immediately deployable primary miner; the S1b
   winning arm as the second miner after S1b concludes.
7. (Strengthened same-day by principal) Motivation-conflicting plans are
   abandoned outright in the proposal, not merely demoted: Section 21.4
   (Plan C) is rewritten in place as a deprecated historical summary that
   forbids future attempts; Section 21.2's automatic gate-to-plan bindings
   become advisory, with any plan switch requiring explicit written
   principal approval; Section 22.4's "Gate A fail → Plan C" agent
   instruction is replaced by the S1b mainline instruction. Standing
   declaration (Section 25.1 item 5): unless the principal actively changes
   course by written decision, the project follows MIRAGE-KAN.

## What is preserved (unchanged invariants)

- Gate A v1 remains sealed as a valid `scientific_fail`. No reopening,
  no reruns, no reinterpretation. Seeds 1729/2718/31415 stay retired.
- All incident reports and custody records remain in force.
- Graph work remains locked; its unlock conditions (S2 value, S3 redundancy
  evidence) are unchanged.
- Statistical discipline (§11.2, §15.10), mechanism cards (§16), matched
  black-box controls, Boost-Sequential and permutation baselines are
  unchanged; the blind interpretability review (§15.9) is strengthened into
  a release gate.
- The honesty clause: interpretability claims may not mask statistically
  significantly worse performance; any such trade-off must be reported as a
  result.

## Consequences

1. Editing `KAN_Alpha_PR.md` intentionally invalidates the proposal hash
   pinned in `prereg/s2_plan_c_vertical_v1.lock.json`
   (was `1880ccf174c3c691f4296eec52b3536d651421f0f0b81bc2d4031d6bc5cdc41a`).
   `verified_s2_identities` will now fail closed on the old S2a protocol.
   This was deliberate, but an already-entitled process was still in flight:
   it read 2016–2021 mining labels and terminalized after the edit. No backtest
   arm or development-test result was opened. The terminal run is retained as
   superseded custody evidence and cannot contribute v2 candidates or claims.
2. S2a must be re-preregistered as v2 (KAN-centered miner pool,
   non-inferiority criteria per Section 25.4) with a fresh lock chain before
   any execution.
3. The old `prereg/s2_plan_c_vertical_v1.md` and its lock are retained
   unmodified as historical records.
4. Custody: the byte-exact pre-edit proposal (sha256
   `1880ccf174c3c691f4296eec52b3536d651421f0f0b81bc2d4031d6bc5cdc41a`, the
   version pinned by the consumed Gate A v1 locks and the superseded S2a v1
   lock) is archived at
   `governance/archives/KAN_Alpha_PR_v2026-07-16_sha1880ccf1.md`. The
   reconstruction was hash-verified against the pinned value, so every
   sealed lock reference remains resolvable on disk.
5. The Gate A v1 machinery (`experiments/gate_a/matrix.py`) verifies the
   live proposal against the consumed v1 preregistration lock and therefore
   now fails closed with "proposal authority hash mismatch" in every mode.
   This is correct: v1 is consumed and its authority superseded. The three
   affected tests are rewritten to assert this refusal and to verify the v1
   lock against the archived proposal; behavioral smoke coverage returns
   with the S1b protocol machinery.
6. Next actions bound to this decision: draft S1b preregistration v1; draft
   S2a preregistration v2; implement the pure-symbolic KAN miner for S2.

## Shelved by this decision

- Plan C's "KAN de-centered, graph-as-protagonist" narrative (§21.4) — the
  evidence discipline of Plan C is retained; only the narrative and miner
  hierarchy change.
- Typed GP/SR as the production mining path (kept as baseline arm only).
- MLP as an auxiliary miner (kept as evaluation control only).
