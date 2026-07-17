# Frozen-seed red-test execution incident

Status: **adjudicated: invalidated before test opening**. Classification: `pre-test partial scientific attempt, invalidated`. Test opening consumed: `false`. Eligible for a clean rerun after audit approval: `true`. This report is append-only evidence; the temporary artifacts below must not be deleted or treated as scientific results.

## Facts and timeline

- At approximately `2026-07-16T22:09:59Z`, a TDD red test intended to prove that the programmatic scientific API rejects a non-default artifact root invoked `run_gate_a_matrix` before that guard existed.
- The invoked test was `tests/experiments/test_gate_a_matrix.py::test_matrix_modes_reject_scientific_overrides_and_frozen_smoke_seeds` within the command below. Pytest had not advanced to the other selected tests.
- Exact command: `PYTHONPATH=src /usr/bin/python -W error -m pytest -q tests/experiments/test_gate_a_matrix.py::test_matrix_modes_reject_scientific_overrides_and_frozen_smoke_seeds tests/experiments/test_gate_a_matrix.py::test_matrix_run_id_is_one_safe_filename_segment tests/experiments/test_gate_a_matrix.py::test_implementation_lock_timestamp_is_truthful`
- The test used valid frozen scientific settings with run ID `fixed_project_artifacts` and a pytest temporary artifact base. It generated seed `1729` data and entered E1 training.
- The process was interrupted with `Ctrl-C` after approximately 15 seconds. Pytest reported `KeyboardInterrupt` in `models.py`; no test completed.
- Exact retained root: `/tmp/pytest-of-root/pytest-134/test_matrix_modes_reject_scien0/s1_gate_a_scientific/fixed_project_artifacts`
- No project-owned `artifacts/s1_gate_a_scientific/` directory was created.

## Scope and evidence boundary

- A complete generated data publication exists for seed `1729`, including train, validation, and test split arrays. The presence of `test.npz` reflects pre-training data materialization; it must not be described as absence of test data generation.
- One E1 selected checkpoint and one E1 console log exist. The console log is a validation/training log and was not semantically inspected during incident forensics.
- No training manifest was published before interruption.
- No `*.test_once.claim`, opening manifest, prediction artifact, receipt, per-seed metric file, Gate report, or top matrix success manifest exists under the retained root.
- Therefore no test-opening claim or prediction/test-metric computation is evidenced. Whether this partial execution changes governance state is reserved to the main agent; this report does not modify preregistration state fields.

## Retained file inventory

| Path | Bytes | mtime UTC | SHA-256 |
|---|---:|---|---|
| `/tmp/pytest-of-root/pytest-134/test_matrix_modes_reject_scien0/s1_gate_a_scientific/fixed_project_artifacts/checkpoints/E1/seed_1729_main_e1/selected.pt` | 3370 | `2026-07-16T22:10:07.311319252Z` | `37d5f1489b170f5709ce097e0672eb05f2e712e4414e082c1d5a93cbfdcc32bf` |
| `/tmp/pytest-of-root/pytest-134/test_matrix_modes_reject_scien0/s1_gate_a_scientific/fixed_project_artifacts/data/seed_1729/arrays/scaler.npz` | 1818 | `2026-07-16T22:10:00.290208946Z` | `1db4abe86db95920a439fefed7518427398c2863535dea01b228294908ad4234` |
| `/tmp/pytest-of-root/pytest-134/test_matrix_modes_reject_scien0/s1_gate_a_scientific/fixed_project_artifacts/data/seed_1729/arrays/test.npz` | 2307930 | `2026-07-16T22:10:00.287208899Z` | `60af1ca6a6c5f8f8d8571d223cd08a926ed75f002312219f9f8a84edce28901d` |
| `/tmp/pytest-of-root/pytest-134/test_matrix_modes_reject_scien0/s1_gate_a_scientific/fixed_project_artifacts/data/seed_1729/arrays/train.npz` | 4290394 | `2026-07-16T22:10:00.269208616Z` | `eff6d2b556133a58afe03b779912beb49d6985fe40311b5a7804c2e4fa920316` |
| `/tmp/pytest-of-root/pytest-134/test_matrix_modes_reject_scien0/s1_gate_a_scientific/fixed_project_artifacts/data/seed_1729/arrays/validation.npz` | 1316698 | `2026-07-16T22:10:00.279208773Z` | `1c7139b803197babccc2b1c3331ff82511ffa08c711d4531bfc2d1c7dad40aa8` |
| `/tmp/pytest-of-root/pytest-134/test_matrix_modes_reject_scien0/s1_gate_a_scientific/fixed_project_artifacts/data/seed_1729/manifest.json` | 2456 | `2026-07-16T22:10:00.290208946Z` | `c8b653fff5b898dfa2d62869aa8573b5df5afc08b181f83de61f5be1e02a0ab5` |
| `/tmp/pytest-of-root/pytest-134/test_matrix_modes_reject_scien0/s1_gate_a_scientific/fixed_project_artifacts/ledgers/matrix_events.jsonl` | 123 | `2026-07-16T22:09:59.027189103Z` | `07783e4f4939c87ec47321b84eb4916bc47a0bd2b5909bfd039f2285774539f8` |
| `/tmp/pytest-of-root/pytest-134/test_matrix_modes_reject_scien0/s1_gate_a_scientific/fixed_project_artifacts/ledgers/training_logs/seed_1729_main_e1/console.log` | 8594 | `2026-07-16T22:10:12.004392982Z` | `436e578f6406fcaba96060f391ff9a8090d27e0afccdd1171e3aefe5940f0d5c` |

## Root cause

The red test used the real scientific runner as the failing seam. Because rejection of a non-default scientific artifact base had not yet been implemented, the expected exception did not occur and the test crossed into the real generator/trainer. The test lacked sentinel monkeypatches that would have made any generator, root preparation, or training call fail immediately without frozen-seed work.

## Corrective actions

1. Validate the fixed scientific artifact base immediately after settings validation and before seal reads, root creation, data generation, or training.
2. Replace the unsafe red test with monkeypatched sentinels for root preparation and data generation, and assert neither sentinel is called.
3. Keep all subsequent execution limited to validation-only tests and non-frozen fresh smoke seeds until the main agent adjudicates governance impact.
4. Preserve the pytest temporary root and this incident report; do not delete or relabel either.
5. Do not edit `scientific_results_observed` or related preregistration fields without main-agent direction.
