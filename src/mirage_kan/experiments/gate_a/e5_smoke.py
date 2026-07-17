"""Train/validation-only non-scientific smoke for the Gate A E5 arm."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from .data import generate_gate_a_replication
from .e5 import E5SearchSettings, save_e5_search, search_e5


def _verify_seal(root: Path) -> dict[str, Any]:
    lock = json.loads((root / "prereg/s1_gate_a_v0.lock.json").read_text())
    for path_key, hash_key in (
        ("protocol_path", "protocol_sha256"),
        ("config_path", "config_sha256"),
    ):
        path = root / lock[path_key]
        if hashlib.sha256(path.read_bytes()).hexdigest() != lock[hash_key]:
            raise RuntimeError(f"sealed file hash mismatch: {path}")
    loaded = yaml.safe_load((root / lock["config_path"]).read_text())
    if not isinstance(loaded, dict):
        raise ValueError("sealed Gate A config must be a mapping")
    return loaded


def run_e5_smoke(
    sealed_config: Mapping[str, Any],
    *,
    run_id: str,
    seed: int,
    candidate_budget: int,
    artifact_root: Path | str = "artifacts/s1_gate_a_e5_smoke",
) -> Path:
    """Run a fresh reduced E5 search without accessing a test-dataset field."""
    frozen_seeds = {int(value) for value in sealed_config["seeds"]}
    if seed in frozen_seeds:
        raise ValueError("E5 smoke seed must not equal a frozen scientific seed")
    if not 1 <= candidate_budget < int(
        sealed_config["symbolic_search"]["max_distinct_valid_ast_evaluations"]
    ):
        raise ValueError("E5 smoke candidate budget must be positive and reduced")
    smoke_config = deepcopy(dict(sealed_config))
    smoke_config["panel"]["assets"] = 4
    smoke_config["panel"]["split_dates"] = {
        "train": 32,
        "validation": 8,
        "test": 8,
    }
    replication = generate_gate_a_replication(smoke_config, seed)
    result = search_e5(
        replication.train.features.numpy(),
        replication.train.noisy_target.to_numpy(),
        replication.validation.features.numpy(),
        replication.validation.clean_truth.to_numpy(),
        source_names=replication.feature_names,
        settings=E5SearchSettings(
            max_distinct_valid_evaluations=candidate_budget,
            seed=seed,
        ),
    )
    resolved_config = json.dumps(
        smoke_config, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return save_e5_search(
        result,
        Path(artifact_root) / "runs" / run_id,
        metadata={
            "smoke_scope": "train_validation_only",
            "scientific_evidence": False,
            "test_evaluated": False,
            "seed": int(seed),
            "candidate_budget": int(candidate_budget),
            "resolved_smoke_config_sha256": hashlib.sha256(
                resolved_config
            ).hexdigest(),
            "generated_content_sha256": replication.provenance["content_sha256"],
            "accessed_dataset_fields": ["train", "validation"],
        },
    )


def main() -> None:
    """Run the categorized E5 smoke from the project root."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=8675309)
    parser.add_argument("--candidate-budget", type=int, default=96)
    parser.add_argument(
        "--artifact-root", default="artifacts/s1_gate_a_e5_smoke"
    )
    args = parser.parse_args()
    manifest = run_e5_smoke(
        _verify_seal(Path.cwd()),
        run_id=args.run_id,
        seed=args.seed,
        candidate_budget=args.candidate_budget,
        artifact_root=args.artifact_root,
    )
    print("e5_smoke_complete=true scope=train_validation_only test_evaluated=false")
    print(f"manifest={manifest}")


if __name__ == "__main__":
    main()
