# S2a v6 authoritative Qlib tracking root

This directory is reserved for the lineage-corrective v6 development run. The
v6 launcher changes into this directory before importing Qlib, so Qlib's local
experiment manager resolves its file store here.

All predecessor tracking trees are quarantined and must never be read as v6
scientific input or appended by v6. Raw MLflow files remain non-authoritative;
only atomically published MIRAGE-KAN evaluation bundles may support decisions.
