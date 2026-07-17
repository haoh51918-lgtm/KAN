# Runtime Logs

Store raw stdout/stderr under one no-replace directory per run. Human summaries, metrics, predictions, and factor libraries belong in their dedicated top-level directories instead.

The initial S0 run preserved its Qlib/MLflow runtime record under `evaluations/runtime/seed_ast_v1/` but did not capture a separate console transcript. This limitation is recorded honestly; no retrospective log is fabricated.
