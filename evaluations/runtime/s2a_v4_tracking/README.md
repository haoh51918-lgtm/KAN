# S2a v4 tracking runtime

Run the S2a v4 development process with this directory as its working
directory. Qlib 0.9.7 derives its default file-backed MLflow tracking URI at
first import, so this keeps runtime tracking state under the categorized
evaluation tree instead of the project root.

The development launcher must set `MLFLOW_ALLOW_FILE_STORE=true` before any
Qlib import and verify that Qlib's configured experiment-manager URI resolves
to the `mlruns` directory created here.
