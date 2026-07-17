# Project-wide root-cause audit — 2026-07-17

## Conclusion

The stall was not caused by one bad runtime call. It was a coupled system failure:

1. multiple documents simultaneously claimed authority and disagreed about the current stage;
2. scientific irreversibility was conflated with ordinary software correctness;
3. execution locks captured dynamic documents and invocation details that do not determine scientific results;
4. EvoSci changed project identity after `git init`, hiding the relevant memory while leaving stale instructions in the old identity;
5. the agent responded to infrastructure failures by generating successor protocols and more proof obligations instead of fixing the shared seam;
6. the research plan optimized for audit completeness and “reviewer-proof” coverage before establishing a sufficiently expressive factor-search system.

The plugin design amplified the problem, but the agent’s execution policy turned that pressure into a full dead loop. The failure is therefore both architectural and behavioral.

## Evidence

### Document conflict

- `README.md` and `research_request.md` called `KAN_Alpha_PR.md` the sole authority.
- The Living Manual and the review-adoption decision said the proposal was a historical baseline and the Living Manual was current.
- README/plans/tracker described graph work as locked and S2a v4/v6 as current, while the graph-unlock decision and later physical artifacts showed otherwise.
- `KAN_Alpha_PR.md` contained an “agent execution contract” and “non-negotiable” directives even though it began as an idea draft.

### Physical S2a state

- v8’s development opening exists and is consumed.
- Five arm-consumption records exist.
- Five Quanta evaluation trees exist.
- Decision assembly terminalized with `implementation runtime identity mismatch`.
- Arm manifests explicitly deny `scientific_result` and formal promotion.

Thus v8 was not “never opened”; it was opened, evaluated, and then invalidated as scientific evidence by a software identity check.

### Over-broad lock

The v8 implementation lock covered 90 files, including `Review-from-claude.md` and the Living Manual. Appending the second review changed only the review file, yet `lock-verify` failed with `predecessor custody hash mismatch`. Runtime identity also serialized raw `PYTHONPATH` and full `sys.path`, so a legitimate launcher/cwd change could cause `implementation runtime identity mismatch` without any change to code, data, dependencies, or GPU.

### EvoSci identity and memory

- Before Git initialization, the workspace resolved to project `P-fb6a3214ec05a54d` and accumulated 19 observations.
- After `git init`, the same workspace resolved to `P-80832d108b90e836`, whose profile and observation set were empty.
- Old observations asserted that the proposal was sole authority, Plan C was active, and graph work was locked.

The resolver recomputed identity from mutable repository state instead of honoring `.evosci/project.json`. This both erased useful continuity and made later recovery dangerous because the old memory contained stale commands.

### Official-source verification

The adapter was checked against EvoScientist 0.2.2 at pinned commit `e8399b7c94e8f97b8935ca97f8efe5eec61c8159`, the official 2026-07-17 main head `06a9511bdd57ba9c583737cf923b66343ad8ec31`, and EvoSkills main head `29e2c67f12858829ad0900645432b340c3f77522`.

- The pinned and current-main EvoScientist core files have identical hashes for the main prompt, memory middleware, project resolver, scheduler, AutoSkills candidate/proposal tools, Feishu channel, and settings.
- The local upstream prompt and memory snapshots are byte-exact. The official prompt is genuinely adaptive, but also mandates code-mode selection, encourages delegation/reflection, requires a final report for the end-to-end workflow, and defines a reviewer-complete stopping checklist.
- The official memory middleware genuinely injects a required observation preflight before workspace inspection. The official scheduler launches memory/linking work asynchronously after source runs; it does not require draining old work before the next scientific task.
- The official project resolver uses remote, then Git root, then path. It has no activation marker, so `git init` can change identity. Marker-first stability is an intentional adapter correction.
- The installed EvoSkills `research-ideation`, `experiment-pipeline`, `experiment-craft`, `evo-memory`, and paper workflows match official main byte-for-byte. `experiment-pipeline` really has strict four-stage attempt budgets and failure logging; its own routing makes it unsuitable for isolated code/runtime bugs, which belong to direct debugging or `experiment-craft`.
- Official chat channels are disabled by default (`channel_enabled=""`). The adapter's former “activate Feishu whenever configured” rule was not upstream behavior.

Primary sources: [EvoScientist prompt](https://github.com/EvoScientist/EvoScientist/blob/e8399b7c94e8f97b8935ca97f8efe5eec61c8159/EvoScientist/prompts.py), [memory middleware](https://github.com/EvoScientist/EvoScientist/blob/e8399b7c94e8f97b8935ca97f8efe5eec61c8159/EvoScientist/middleware/memory.py), [project resolver](https://github.com/EvoScientist/EvoScientist/blob/e8399b7c94e8f97b8935ca97f8efe5eec61c8159/EvoScientist/memory/project.py), [memory scheduler](https://github.com/EvoScientist/EvoScientist/blob/e8399b7c94e8f97b8935ca97f8efe5eec61c8159/EvoScientist/memory/scheduler.py), and [official EvoSkills](https://github.com/EvoScientist/EvoSkills/tree/main/skills).

### Responsibility split

| Layer | Contribution to the loop | Disposition |
|---|---|---|
| Upstream EvoScientist | Mandatory memory preflight and reviewer-complete workflow create real control-plane pressure. | Retain as source context, but do not copy mandatory timing into Codex. |
| Codex adapter | Deferred memory was drained before work; Feishu was auto-bound when configured; exact-CLI claims hid intentional differences. | Memory is targeted/batched, Feishu is explicit opt-in, and semantic differences are now documented and tested. |
| EvoSkills routing | A strict full experiment pipeline was available and was treated as generally applicable. | Route only planned multi-stage experiments to it; isolated bugs use direct debugging/experiment-craft. |
| Project documents | PR, manual, README, plans, tracker, review, and memory competed as authorities. | One behavior entry, one current state, one action queue; PR/review/history are non-authoritative. |
| Agent execution | Each infrastructure defect became a scientific custody event and successor protocol. | Reproduce/red-test/fix the same reversible seam; no successor without changed scientific semantics. |

## Root cause model

```text
conflicting authority documents
        +
mandatory memory/reflection/audit preflights
        +
locks over docs and process-local state
        +
no stop rule for repeated infrastructure failures
        ↓
software bug is classified as a scientific custody event
        ↓
new protocol → new lock → new launcher/receipt seam
        ↓
larger failure surface and another terminal successor
        ↓
less time for factor space, operators, S1b, graph, and real analysis
```

The underlying conceptual error was treating “more proof” as monotonic progress. Proof machinery has its own defect rate and opportunity cost. Once it becomes larger than the scientific system, adding controls can reduce rather than increase scientific reliability.

## Repairs applied

- Added a single project behavior contract in root `AGENTS.md`, made the Living Manual the single current-state source, and made `plans/todos.md` the single action queue.
- Reclassified the proposal as an idea draft and the Claude review as advisory.
- Distilled README/research request/current plans; converted the tracker to history; merged and deleted the redundant `plans/success_criteria.md`; prohibited reflexive v9 creation.
- Excluded dynamic control documents and invocation-only variables from future execution identities.
- Added regression tests proving that review edits and launcher-path changes do not alter execution identity.
- Made EvoSci project identity marker-first and added a regression test across `git init`.
- Changed EvoSci memory retrieval and maintenance from a mandatory blocking preflight to targeted, non-blocking support work.
- Made routine EvoSci `init` non-capturing; durable TURN memory capture now requires explicit `--arm-turn`, with default-off and opt-in hook regression tests.
- Made delegation optional, defaulted scoped code to Lite, routed isolated bugs away from `experiment-pipeline`, and made Feishu activation/reporting explicit opt-in.
- Restored `/root/plugins/evosci-core` as the maintained local plugin source, applied a cachebuster, reinstalled the plugin, and verified the installed cache against that source.
- Restored the original EvoSci project identity and superseded stale authority/route memories rather than deleting history.
- Updated future publication identities and authority guards to use the proposal's `idea_draft` role while retaining compatibility with frozen historical locks.

## Remaining scientific risks

- The present E3 space appears too small and structurally weak; exact enumeration must quantify this before more training.
- Seven strict factors are insufficient evidence for a graph coordinator; graph prototypes must not be judged on an artificially tiny pool.
- S1b may still fail, in which case the defensible result is an identifiability boundary/negative result rather than another threshold adjustment.
- Quanta metrics from v8 are useful only for diagnosing pipeline behavior; treating them as results would contaminate the evidence boundary.
- The repository has no baseline Git commit, so change provenance remains weak until the user chooses an initial commit strategy.

## Verification

- Project suite: `399 passed` in the pinned runtime; the 1,045 warnings are the existing pandas `FutureWarning` from DSL support-mask filling.
- Focused authority/lock/CLI regression set: `68 passed`; project Ruff check passed.
- EvoSci plugin: `75 passed`; plugin and all three bundled skills validate; plugin Ruff check passed.
- The maintained local plugin source is `/root/plugins/evosci-core`, installed as `0.1.0+codex.20260717172323`; its installed cache is byte-identical apart from excluded test caches.
- The only agent-instruction entry found in the project is root `AGENTS.md`; no competing `CLAUDE.md`, `CODEX.md`, `.cursorrules`, Cursor rule, or Copilot instruction file exists.

## Prevention rules

- One current-state document, many historical evidence documents.
- One failure class, one owner: software bugs go to tests/code; scientific uncertainty goes to experiments; data-opening decisions go to preregistration.
- Governance must be cheaper than the risk it removes.
- Every control mechanism needs a stop condition and a falsifiable reason to exist.
- When the scientific queue has not advanced during a working session, stop adding control-plane work and return to the smallest unresolved scientific question.
