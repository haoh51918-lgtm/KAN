# S2a v5 authoritative Qlib tracking root

This directory is reserved for the calendar-corrective v5 development run.
The v5 launcher changes into this directory before importing Qlib, so Qlib's
authoritative file-store URI resolves to the local `mlruns/` child.

The predecessor `evaluations/runtime/s2a_v4_tracking` tree is quarantined and
must never be read as v5 scientific input or appended by v5. Raw MLflow default
URI diagnostics are non-authoritative; `MLFLOW_TRACKING_URI` remains unset.
