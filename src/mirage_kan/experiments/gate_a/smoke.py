"""Tiny non-scientific E1/C6 harness smoke entry point."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import yaml

from .data import generate_gate_a_replication, save_gate_a_replication
from .models import CAPACITY_SPEC, FreeSplineKAN, MatchedMLP
from .training import TrainingSettings, evaluate_test_once, train_and_select


def _verify_seal(root: Path) -> dict:
    lock = json.loads((root / "prereg/s1_gate_a_v0.lock.json").read_text())
    for path_key, hash_key in (
        ("protocol_path", "protocol_sha256"),
        ("config_path", "config_sha256"),
    ):
        path = root / lock[path_key]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != lock[hash_key]:
            raise RuntimeError(f"sealed file hash mismatch: {path}")
    return yaml.safe_load((root / lock["config_path"]).read_text())


def _record(path: Path, message: str) -> None:
    print(message, flush=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(message + "\n")


def main() -> None:
    """Run a tiny smoke without touching the frozen three-seed scientific matrix."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("E1", "C6"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--artifact-root", default="artifacts/s1_gate_a_smoke")
    parser.add_argument("--log-root", default="logs")
    args = parser.parse_args()

    root = Path.cwd()
    sealed = _verify_seal(root)
    smoke_config = deepcopy(sealed)
    smoke_config["panel"]["assets"] = 4
    smoke_config["panel"]["split_dates"] = {
        "train": 32,
        "validation": 8,
        "test": 8,
    }
    replication = generate_gate_a_replication(smoke_config, args.seed)
    artifact_root = Path(args.artifact_root)
    data_manifest = save_gate_a_replication(
        replication, artifact_root / "data" / args.run_id
    )
    settings = TrainingSettings.from_config(sealed, seed=args.seed)
    settings = replace(
        settings,
        max_steps=args.steps,
        batch_size=min(settings.batch_size, len(replication.train.features)),
        validation_interval_steps=1,
        early_stopping_patience_validations=args.steps,
    )
    model = (
        FreeSplineKAN()
        if args.arm == "E1"
        else MatchedMLP(CAPACITY_SPEC, initialization_seed=args.seed)
    )
    run = train_and_select(
        model,
        replication.train,
        replication.validation,
        settings,
        arm=args.arm,
        run_id=args.run_id,
        data_manifest_path=data_manifest,
        artifact_root=artifact_root,
        log_root=args.log_root,
        device=args.device,
    )
    evaluate_test_once(run, model, replication.test)
    _record(run.console_log_path, f"smoke_complete=true arm={args.arm}")
    _record(run.console_log_path, f"manifest={run.manifest_path}")


if __name__ == "__main__":
    main()
