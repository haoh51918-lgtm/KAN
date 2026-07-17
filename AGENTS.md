# MIRAGE-KAN agent contract

This file controls agent behavior in this workspace. It is intentionally short: the project exists to do scientific research, not to grow a governance system.

## Authority

Use this order when instructions conflict:

1. the latest explicit instruction from the user/principal;
2. this `AGENTS.md` for behavior and `docs/research/MIRAGE_KAN_LIVING_MANUAL.md` for current scientific state and route;
3. an active preregistration, only for the experiment it freezes;
4. code, tests, manifests, and raw artifacts as evidence of what actually happened;
5. decisions, incidents, reports, and reviews as dated historical context.

`KAN_Alpha_PR.md` is an idea draft and hypothesis catalogue, not a constitution or execution contract. `Review-from-claude.md` is valuable advisory analysis, not authority. Historical files never become current instructions merely because they contain words such as “must”, “locked”, or “sole authority”.

## Science first

- Spend the dominant share of effort on methods, experiments, analysis, and falsifiable evidence.
- Prefer the smallest experiment that distinguishes scientific hypotheses. Do not add a receipt, audit, schema, lock, review, or successor protocol unless it covers a concrete risk not already covered by a test or artifact.
- Treat reversible software work and irreversible statistical access separately. Debugging, unit tests, synthetic data, train-only development, exact enumeration, profiling, and interface prototypes are reversible and do not require a new scientific protocol.
- A new preregistration is needed only when a scientific hypothesis, data role/opening, seed set, success threshold, or confirmatory analysis changes.
- An implementation failure is a software bug first. Reproduce it, write a failing test, fix it, and rerun the same reversible path. Do not create a successor protocol by reflex.
- After two consecutive infrastructure failures at the same seam, stop adding patches and redesign that seam. After two non-improving scientific iterations, revisit the scientific hypothesis.

## Evidence and documentation

- Frozen preregistrations, openings, manifests, results, and incident records are immutable history.
- Current state is written once in the Living Manual. The actionable queue is written once in `plans/todos.md`. README and historical trackers link to them instead of copying their contents.
- Update the Living Manual after a material terminal result or principal route decision; update the queue only when next actions change. Do not create a parallel status document.
- Dynamic control documents (`AGENTS.md`, README, Review, Living Manual, plans, trackers, research request) must never be part of an execution lock.
- Execution identity may include source, data, configs, dependency records, device identity, and determinism settings that affect numerical results. It must not include working directory, `sys.path` ordering, `PYTHONPATH`, logging destinations, or report text.

## Engineering and verification

- Read the relevant code and tests before editing. Fix bugs red-first and keep changes surgical.
- Use the project’s pinned runtime for the main suite; `/usr/bin/python` is not the locked project environment.
- Do not install dependencies merely to run a check when the pinned environment already contains them.
- Preserve external QuantaAlpha and data repositories as read-only dependencies.
- External messages, Feishu reports, backup copies, and confirmation-data openings require explicit current authorization; historical requests do not authorize a new send.
