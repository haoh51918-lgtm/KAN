# Evaluation Runtime Records

This directory stores evaluator-owned runtime records that are useful for audit but are not part of an immutable evaluation publication.

- `seed_ast_v1/mlruns/` is the MLflow/Qlib record created by the successful S0 run. It contains training/validation loss histories, predictions, labels, IC series, and run metadata.
- The immutable result summary and cumulative excess-return series remain under `evaluations/s0_vertical_slice/`.

Future evaluation commands must set their MLflow tracking location or move the completed run store here immediately after the run. Raw console output must be captured under `logs/<run_id>/`; the first S0 run did not preserve a separate console transcript, so none is reconstructed after the fact.

