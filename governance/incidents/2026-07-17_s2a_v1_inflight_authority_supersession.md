# Incident: S2a v1 in-flight proposal-authority supersession

- **Date**: 2026-07-17 UTC
- **Classification**: `in_flight_authority_supersession`
- **Run disposition**: `superseded_non_scientific_run`
- **Severity**: governance high; scientific test contamination not observed
- **Status**: contained; v2 prevention required before execution

## What happened

The `s2_plan_c_vertical_v1` process consumed its prospective mining entitlement
under the then-locked proposal and remained active while the live sole proposal
authority was revised to the KAN-first Section 25. The running process checked
the proposal identity before starting, but did not recheck it at scientific-arm
or publication boundaries. It therefore continued for approximately 65 seconds
after the proposal changed and then closed as a terminal protocol failure.

This was not an authorized successor run, a rerun, or a development-test
opening. It was an already-running process whose authority became stale.

## Evidence timeline

| Event | UTC timestamp |
|---|---|
| v1 mining preclaim created | 2026-07-17 01:41:44 |
| v1 256-attempt entitlement consumed | 2026-07-17 01:41:54 |
| live `KAN_Alpha_PR.md` revised | 2026-07-17 01:58:44 |
| KAN-first directive record written | 2026-07-17 01:59:19 |
| v1 terminal ledgers and manifest finalized | 2026-07-17 01:59:48–01:59:49 |
| v1 failure report and decision written | 2026-07-17 02:02:18 |

The sealed v1 data-access ledger records access to discovery/validation `fwd`
labels through 2021-12-31 and loading of the complete publication OHLCV panel.
It records `development_test_outcome_access=false`. No 2022–2025 outcome,
Quanta evaluation, portfolio metric, or development decision was opened.

## Scientific custody

1. All v1 artifacts remain immutable and are not deleted or relabeled.
2. No v1 terminal factor-library directory is publishable or executable.
3. The run is not evidence for or against the current KAN-centered proposal.
4. v1 search/control seeds and its candidate topology are retired from v2
   scientific use. They may be used only in isolated engineering regression
   tests that cannot influence v2 candidate membership or thresholds.
5. Quanta, PIT data, baseline, metric, and execution identities may be reused;
   they are infrastructure rather than v1-selected scientific content.

## Factual correction

The directive record's original consequence text said that no mining labels had
been opened under v1. That statement was true when the proposal edit began but
became false before the directive file was finalized because the in-flight run
had already consumed its entitlement and read 2016–2021 labels. The accurate
boundary is: **v1 mining labels were opened, but no development-test outcome or
Quanta result was opened**.

This correction does not change Section 25's authority or method decision. It
only corrects the custody history. Editing the proposal itself remains reserved
to the principal.

## Required v2 prevention

Before v2 can execute, its implementation must revalidate the live proposal
hash at all of the following boundaries:

1. immediately before first label access;
2. immediately before every scientific/control arm;
3. immediately before factor/control publication;
4. immediately before the one-shot development opening; and
5. immediately before final decision publication.

Any mismatch must terminalize the current run as `superseded_authority`; it may
not publish a scientific result. The v2 protocol must use a new ID, new seeds,
new no-replace paths, and a complete new lock chain.

