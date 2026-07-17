"""Train/validation-only tiny smoke for the Gate A E2/E3/E4 symbolic arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import torch
import yaml

from .data import generate_gate_a_replication, save_gate_a_replication
from .models import FreeSplineKAN, SymbolicKAN, SymbolicResidualKAN
from .posthoc import symbolify_e1_checkpoint
from .symbolic import fidelity_metrics, save_hard_export
from .training import TrainingSettings, train_and_select


def _verify_seal(root: Path) -> dict:
    lock = json.loads((root / "prereg/s1_gate_a_v0.lock.json").read_text())
    for path_key, hash_key in (
        ("protocol_path", "protocol_sha256"),
        ("config_path", "config_sha256"),
    ):
        path = root / lock[path_key]
        if hashlib.sha256(path.read_bytes()).hexdigest() != lock[hash_key]:
            raise RuntimeError(f"sealed file hash mismatch: {path}")
    return yaml.safe_load((root / lock["config_path"]).read_text())


def _validation_fidelity(
    model: torch.nn.Module, hard: torch.nn.Module, features: torch.Tensor, device: str
) -> dict[str, float]:
    target = torch.device(device)
    values = features.to(device=target, dtype=torch.float64)
    model.to(device=target, dtype=torch.float64).eval()
    hard.to(device=target, dtype=torch.float64).eval()
    with torch.no_grad():
        return fidelity_metrics(model(values), hard(values))


def main() -> None:
    """Run one tiny symbolic arm without evaluating the frozen test split."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("E2", "E3", "E4"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--artifact-root", default="artifacts/s1_gate_a_symbolic_smoke")
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
    settings = replace(
        TrainingSettings.from_config(sealed, seed=args.seed),
        max_steps=args.steps,
        batch_size=min(256, len(replication.train.features)),
        validation_interval_steps=1,
        early_stopping_patience_validations=args.steps,
    )

    if args.arm == "E2":
        source_run_id = f"{args.run_id}_e1_source"
        source = FreeSplineKAN()
        source_run = train_and_select(
            source,
            replication.train,
            replication.validation,
            settings,
            arm="E1_for_E2",
            run_id=source_run_id,
            data_manifest_path=data_manifest,
            artifact_root=artifact_root,
            log_root=args.log_root,
            device=args.device,
        )
        source.to(device="cpu", dtype=torch.float64)
        result = symbolify_e1_checkpoint(
            source_run.checkpoint_path,
            source,
            replication.train.features,
            source_names=replication.feature_names,
        )
        validation_fidelity = result.fidelity(replication.validation.features)
        metadata = {
            "smoke_scope": "train_validation_only",
            "source_run_manifest": str(source_run.manifest_path),
            "fit_manifest": result.fit_manifest,
            "validation_fidelity": validation_fidelity,
        }
        hard = result.hard_model
        own_log = Path(args.log_root) / args.run_id / "console.log"
        own_log.parent.mkdir(parents=True, exist_ok=False)
        own_log.write_text(
            f"run={args.run_id} arm=E2 scope=train_validation_only\n"
            f"fit_count={result.fit_count} validation_fidelity={validation_fidelity}\n",
            encoding="utf-8",
        )
    else:
        model = SymbolicKAN() if args.arm == "E3" else SymbolicResidualKAN()
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
        hard = model.harden()
        validation_fidelity = _validation_fidelity(
            model, hard, replication.validation.features, args.device
        )
        metadata = {
            "smoke_scope": "train_validation_only",
            "source_run_manifest": str(run.manifest_path),
            "validation_fidelity": validation_fidelity,
            "residual_energy": (
                model.residual_energy(
                    replication.validation.features.to(
                        device=args.device, dtype=torch.float64
                    )
                )
                .detach()
                .cpu()
                .tolist()
            ),
        }

    manifest = save_hard_export(
        hard,
        artifact_root / "symbolic_exports" / args.arm / args.run_id,
        arm=args.arm,
        metadata=metadata,
    )
    print(f"symbolic_smoke_complete=true arm={args.arm} test_evaluated=false")
    print(f"hard_manifest={manifest}")


if __name__ == "__main__":
    main()
