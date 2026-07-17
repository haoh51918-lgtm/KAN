"""Exact E1--E5+C6 S1 Gate A matrix orchestration."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import yaml

from .controls import permute_dataset_labels, remove_feature_source
from .data import (
    FEATURE_NAMES,
    GateAReplication,
    generate_gate_a_replication,
    save_gate_a_replication,
)
from .e5 import E5SearchSettings, save_e5_search, search_e5
from .metrics import (
    PER_ARM_NUMERIC_KEYS,
    REPORTED_ARMS,
    aggregate_arm_metrics,
    clean_mechanism,
    evaluate_gate_a,
    evaluate_mechanism_shape,
    metric_not_applicable,
    nrmse,
    open_test_matrix_once,
    write_gate_report,
)
from .models import FreeSplineKAN, MatchedMLP, SymbolicKAN, SymbolicResidualKAN
from .posthoc import symbolify_e1
from .promotion import (
    EligibilityResult,
    PromotedPrimitive,
    PromotionResult,
    ResidualShape,
    assess_residual_eligibility,
    extract_residual_shape,
    fit_governed_promotion,
    refit_promoted_hard_model,
    save_no_promotion_status,
    save_promoted_hard_model,
    save_residual_shape,
)
from .symbolic import fidelity_metrics, save_hard_export
from .training import TrainingRun, TrainingSettings, train_and_select

FROZEN_SEEDS = (1729, 2718, 31415)
SEALED_CONFIG = Path("configs/experiments/s1_gate_a_v1.yaml")
SEALED_LOCK = Path("prereg/s1_gate_a_v1.lock.json")
IMPLEMENTATION_LOCK = Path("prereg/s1_gate_a_v1_implementation.lock.json")
KNOWN_INCIDENT_REPORT = Path(
    "governance/incidents/2026-07-16_frozen_seed_red_test_incident.md"
)
KNOWN_INCIDENT_CLEANUP_ADDENDUM = Path(
    "governance/incidents/2026-07-16_frozen_seed_red_test_cleanup_addendum.md"
)
ARTIFACT_SUBDIRECTORIES = (
    "data",
    "checkpoints",
    "models",
    "residual_shapes",
    "controls",
    "predictions",
    "metrics",
    "ledgers",
    "manifests",
    "reports",
)


@dataclass(frozen=True)
class MatrixSettings:
    """Validated scientific or explicitly reduced smoke execution choices."""

    mode: str
    run_id: str
    seeds: tuple[int, ...]
    config: Mapping[str, Any]
    e5_candidate_budget: int
    bootstrap_replicates: int

    def validate(self) -> None:
        if self.mode not in {"scientific", "smoke"}:
            raise ValueError("matrix mode and run ID are required")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", self.run_id) is None:
            raise ValueError("matrix run ID must be one safe filename segment")
        if len(set(self.seeds)) != len(self.seeds) or len(self.seeds) < 2:
            raise ValueError("matrix needs at least two distinct seeds")
        if self.mode == "scientific":
            _verify_implementation_lock()
            if self.seeds != FROZEN_SEEDS:
                raise ValueError("scientific matrix must use the three sealed seeds")
            if self.e5_candidate_budget != 12_000 or self.bootstrap_replicates != 1000:
                raise ValueError("scientific matrix budgets must equal the sealed settings")
            if dict(self.config) != _load_sealed_config():
                raise ValueError("scientific matrix config differs from the sealed config")
        elif any(seed in FROZEN_SEEDS for seed in self.seeds):
            raise ValueError("smoke mode cannot use a frozen scientific seed")
        if self.e5_candidate_budget < 1 or self.bootstrap_replicates < 1:
            raise ValueError("matrix search and bootstrap budgets must be positive")


def _load_sealed_config() -> dict[str, Any]:
    value = yaml.safe_load(SEALED_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("sealed Gate A config is not a mapping")
    return value


def scientific_matrix_settings(
    matrix_id: str,
    *,
    overrides: Mapping[str, Any] | None = None,
    seeds: Sequence[int] | None = None,
) -> MatrixSettings:
    """Return exact sealed scientific settings; every override is forbidden."""
    if overrides is not None or seeds is not None:
        raise ValueError("scientific mode does not accept overrides or fresh seeds")
    settings = MatrixSettings(
        "scientific", matrix_id, FROZEN_SEEDS, _load_sealed_config(), 12_000, 1000
    )
    settings.validate()
    return settings


def smoke_matrix_settings(
    run_id: str,
    *,
    seeds: Sequence[int],
    assets: int,
    burn_in_dates: int,
    train_dates: int,
    validation_dates: int,
    test_dates: int,
    max_steps: int,
    batch_size: int,
    e5_candidate_budget: int,
    bootstrap_replicates: int,
) -> MatrixSettings:
    """Build an explicitly reduced non-scientific config with fresh seeds only."""
    config = copy.deepcopy(_load_sealed_config())
    explicit_positive = (
        assets,
        burn_in_dates,
        train_dates,
        validation_dates,
        test_dates,
        max_steps,
        batch_size,
        e5_candidate_budget,
        bootstrap_replicates,
    )
    if any(value < 1 for value in explicit_positive):
        raise ValueError("all explicit smoke reductions must be positive")
    config["panel"]["assets"] = int(assets)
    config["panel"]["burn_in_dates"] = int(burn_in_dates)
    config["panel"]["split_dates"] = {
        "train": int(train_dates),
        "validation": int(validation_dates),
        "test": int(test_dates),
    }
    config["training"]["max_steps"] = int(max_steps)
    config["training"]["batch_size"] = int(batch_size)
    config["training"]["validation_interval_steps"] = 1
    config["training"]["early_stopping_patience_validations"] = max(1, int(max_steps))
    settings = MatrixSettings(
        "smoke",
        run_id,
        tuple(int(seed) for seed in seeds),
        config,
        int(e5_candidate_budget),
        int(bootstrap_replicates),
    )
    settings.validate()
    return settings


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seal_snapshot() -> dict[str, Any]:
    lock = json.loads(SEALED_LOCK.read_text(encoding="utf-8"))
    proposal_path = Path(lock["proposal_authority"])
    actual_protocol = _sha256(Path(lock["protocol_path"]))
    actual_config = _sha256(Path(lock["config_path"]))
    snapshot = {
        "lock_path": str(SEALED_LOCK),
        "lock_sha256": _sha256(SEALED_LOCK),
        "proposal_path": str(proposal_path),
        "proposal_sha256": _sha256(proposal_path),
        "protocol_sha256": actual_protocol,
        "config_sha256": actual_config,
        "protocol_matches": actual_protocol == lock["protocol_sha256"],
        "config_matches": actual_config == lock["config_sha256"],
    }
    if not snapshot["protocol_matches"] or not snapshot["config_matches"]:
        raise ValueError("sealed Gate A protocol/config hash mismatch")
    if lock["proposal_authority"] != "KAN_Alpha_PR.md":
        raise ValueError("proposal authority is not KAN_Alpha_PR.md")
    if _sha256(proposal_path) != lock["proposal_sha256"]:
        raise ValueError("proposal authority hash mismatch")
    return snapshot


def _implementation_snapshot() -> dict[str, Any]:
    model_specs = sorted(Path("configs/model_specs").glob("s1_gate_a_*_v0.json"))
    if len(model_specs) != 4:
        raise ValueError("implementation snapshot requires exactly four S1 model specs")
    paths = sorted(Path("src/mirage_kan").rglob("*.py")) + model_specs + [
        Path("configs/experiments/s1_gate_a_matrix_runner_v0.json"),
        Path("pyproject.toml"),
    ]
    records = {str(path): _sha256(path) for path in paths}
    aggregate = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"aggregate_sha256": aggregate, "files": records}


def _verify_implementation_lock() -> dict[str, Any]:
    if not IMPLEMENTATION_LOCK.is_file():
        raise FileNotFoundError("scientific implementation lock is missing")
    lock = json.loads(IMPLEMENTATION_LOCK.read_text(encoding="utf-8"))
    if lock.get("created_before_science") is not True:
        raise ValueError("implementation lock was not declared before science")
    if lock.get("scientific_results_observed") is not False:
        raise ValueError("implementation lock scientific-results state is invalid")
    snapshot = _implementation_snapshot()
    if lock.get("snapshot") != snapshot:
        raise ValueError("implementation differs from the scientific implementation lock")
    model_specs = {
        path: digest
        for path, digest in snapshot["files"].items()
        if path.startswith("configs/model_specs/s1_gate_a_")
    }
    if lock.get("model_specs") != model_specs:
        raise ValueError("implementation lock model-spec index differs from snapshot")
    matrix_spec_path = "configs/experiments/s1_gate_a_matrix_runner_v0.json"
    if lock.get("matrix_spec") != {
        "path": matrix_spec_path,
        "sha256": snapshot["files"][matrix_spec_path],
    }:
        raise ValueError("implementation lock matrix-spec identity is invalid")
    actual_max_mtime_ns = max(
        Path(path).stat().st_mtime_ns for path in snapshot["files"]
    )
    if lock.get("max_locked_file_mtime_ns") != actual_max_mtime_ns:
        raise ValueError("implementation lock maximum file mtime is invalid")
    recorded_max_mtime = datetime.fromisoformat(lock["max_locked_file_mtime_utc"])
    created_at = datetime.fromisoformat(lock["created_at_utc"])
    if recorded_max_mtime.tzinfo is None or created_at.tzinfo is None:
        raise ValueError("implementation lock timestamps must include UTC offsets")
    recorded_max_mtime_ns = (
        int(recorded_max_mtime.timestamp()) * 1_000_000_000
        + recorded_max_mtime.microsecond * 1_000
    )
    if not 0 <= actual_max_mtime_ns - recorded_max_mtime_ns < 1_000:
        raise ValueError("implementation lock maximum file mtime UTC is invalid")
    created_at_ns = (
        int(created_at.timestamp()) * 1_000_000_000
        + created_at.microsecond * 1_000
    )
    if created_at_ns < actual_max_mtime_ns:
        raise ValueError("implementation lock timestamp predates a locked file")
    expected_incidents = [
        {
            "report_path": str(KNOWN_INCIDENT_REPORT),
            "report_sha256": _sha256(KNOWN_INCIDENT_REPORT),
            "classification": "pre-test partial scientific attempt, invalidated",
            "test_opened": False,
            "eligible_for_clean_rerun": True,
        }
    ]
    if lock.get("known_preformal_incidents") != expected_incidents:
        raise ValueError("implementation lock preformal incident ledger is invalid")
    expected_addenda = [
        {
            "path": str(KNOWN_INCIDENT_CLEANUP_ADDENDUM),
            "sha256": _sha256(KNOWN_INCIDENT_CLEANUP_ADDENDUM),
            "relation": "custody update to known_preformal_incidents[0]",
        }
    ]
    if lock.get("governance_addenda") != expected_addenda:
        raise ValueError("implementation lock governance addenda are invalid")
    return {"path": str(IMPLEMENTATION_LOCK), "sha256": _sha256(IMPLEMENTATION_LOCK)}


def _append_event(path: Path, event: str, **values: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {"event": event, **values},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _prepare_root(settings: MatrixSettings, artifact_base: Path | str) -> Path:
    category = (
        "s1_gate_a_scientific"
        if settings.mode == "scientific"
        else "s1_gate_a_matrix_smoke"
    )
    category_root = (Path(artifact_base) / category).resolve()
    root = (category_root / settings.run_id).resolve()
    if root.parent != category_root:
        raise ValueError("matrix run root escapes its artifact category")
    category_root.mkdir(parents=True, exist_ok=True)
    root.mkdir(exist_ok=False)
    for name in ARTIFACT_SUBDIRECTORIES:
        (root / name).mkdir()
    return root


def _training_settings(settings: MatrixSettings, seed: int) -> TrainingSettings:
    return TrainingSettings.from_config(dict(settings.config), seed=seed)


def _train(
    model: torch.nn.Module,
    replication: GateAReplication,
    settings: MatrixSettings,
    *,
    seed: int,
    arm: str,
    suffix: str,
    data_manifest: Path,
    root: Path,
    device: str,
    train_override: Any | None = None,
    validation_override: Any | None = None,
) -> TrainingRun:
    return train_and_select(
        model,
        train_override if train_override is not None else replication.train,
        validation_override if validation_override is not None else replication.validation,
        _training_settings(settings, seed),
        arm=arm,
        run_id=f"seed_{seed}_{suffix}_{arm.lower()}",
        data_manifest_path=data_manifest,
        artifact_root=root,
        log_root=root / "ledgers" / "training_logs",
        device=device,
    )


def _save_null_control(
    root: Path, seed: int, train: Any, validation: Any, evidence: Mapping[str, Any]
) -> Path:
    output = root / "controls" / "null" / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=False)
    arrays = output / "targets.npz"
    with arrays.open("xb") as stream:
        np.savez(
            stream,
            train_clean=train.clean_truth.to_numpy(),
            train_noisy=train.noisy_target.to_numpy(),
            validation_clean=validation.clean_truth.to_numpy(),
            validation_noisy=validation.noisy_target.to_numpy(),
        )
        stream.flush()
        os.fsync(stream.fileno())
    manifest = output / "input_manifest.json"
    _exclusive_json(
        manifest,
        {
            "schema_version": 1,
            "control": "date_block_label_permutation",
            "seed": seed,
            "evidence": dict(evidence),
            "array_path": str(arrays),
            "array_sha256": _sha256(arrays),
            "test_access": False,
            "publication": {"no_replace": True, "manifest_last": True},
        },
    )
    return manifest


def _finalize_null_control(
    root: Path,
    seed: int,
    input_manifest: Path,
    residual_manifest: Path,
    model: SymbolicResidualKAN,
    train: Any,
    validation: Any,
    device: torch.device,
) -> Path:
    combined = torch.cat((train.features, validation.features)).to(device)
    selected = model.selected_source_metadata(combined)
    predictor = _torch_predictor(model, str(device))
    validation_nrmse = nrmse(
        predictor(validation.features.numpy()), validation.clean_truth.to_numpy()
    )
    manifest = root / "controls" / "null" / f"seed_{seed}" / "manifest.json"
    _exclusive_json(
        manifest,
        {
            "schema_version": 1,
            "control": "date_block_label_permutation",
            "seed": seed,
            "selection": selected,
            "validation_permuted_clean_nrmse": validation_nrmse,
            "residual_shape_manifest": str(residual_manifest),
            "input_manifest": str(input_manifest),
            "input_manifest_sha256": _sha256(input_manifest),
            "test_access": False,
            "publication": {"no_replace": True, "manifest_last": True},
        },
    )
    return manifest


def _save_source_removed_control(
    root: Path,
    seed: int,
    train: Any,
    validation: Any,
    source_names: Sequence[str],
) -> Path:
    output = root / "controls" / "source_removed" / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=False)
    arrays = output / "features.npz"
    with arrays.open("xb") as stream:
        np.savez(
            stream,
            train_features=train.features.numpy(),
            validation_features=validation.features.numpy(),
            source_names=np.asarray(source_names, dtype=str),
        )
        stream.flush()
        os.fsync(stream.fileno())
    manifest = output / "input_manifest.json"
    _exclusive_json(
        manifest,
        {
            "schema_version": 1,
            "control": "exact_source_removed",
            "seed": seed,
            "removed_source": "Return(Close,5)",
            "remaining_sources": list(source_names),
            "hidden_reconstruction": False,
            "future_access": False,
            "test_access": False,
            "array_path": str(arrays),
            "array_sha256": _sha256(arrays),
            "publication": {"no_replace": True, "manifest_last": True},
        },
    )
    return manifest


def _finalize_source_removed_control(
    root: Path,
    seed: int,
    input_manifest: Path,
    residual_manifest: Path,
    model: SymbolicResidualKAN,
    train: Any,
    validation: Any,
    remaining_names: Sequence[str],
    device: torch.device,
    mechanism: Mapping[str, Any],
) -> Path:
    combined = torch.cat((train.features, validation.features)).to(device)
    selected = model.selected_source_metadata(combined)
    predictor = _torch_predictor(model, str(device))
    validation_nrmse = nrmse(
        predictor(validation.features.numpy()), validation.clean_truth.to_numpy()
    )
    mechanism_x = np.linspace(-4.0, 4.0, 801)
    mechanism_prediction = predictor(
        np.zeros((len(mechanism_x), len(remaining_names)), dtype=np.float64)
    )
    mechanism_shape_nrmse = nrmse(
        mechanism_prediction, clean_mechanism(mechanism_x, mechanism)
    )
    manifest = (
        root / "controls" / "source_removed" / f"seed_{seed}" / "manifest.json"
    )
    _exclusive_json(
        manifest,
        {
            "schema_version": 1,
            "control": "exact_source_removed",
            "seed": seed,
            "removed_source": "Return(Close,5)",
            "remaining_sources": list(remaining_names),
            "selected_remaining_source": selected,
            "validation_clean_nrmse": validation_nrmse,
            "mechanism_shape_nrmse_with_removed_coordinate_unavailable": mechanism_shape_nrmse,
            "mechanism_input_construction": "all_remaining_standardized_features_held_at_train_median_zero",
            "residual_shape_manifest": str(residual_manifest),
            "input_manifest": str(input_manifest),
            "input_manifest_sha256": _sha256(input_manifest),
            "hidden_reconstruction": False,
            "future_access": False,
            "test_access": False,
            "publication": {"no_replace": True, "manifest_last": True},
        },
    )
    return manifest


def _torch_predictor(model: torch.nn.Module, device: str) -> Callable[[np.ndarray], np.ndarray]:
    target = torch.device(device)
    model.to(device=target, dtype=torch.float64)
    model.eval()

    def predict(matrix: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return (
                model(torch.as_tensor(matrix, dtype=torch.float64, device=target))
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

    return predict


def _numpy_fidelity(soft: np.ndarray, hard: np.ndarray) -> dict[str, float]:
    return fidelity_metrics(
        torch.as_tensor(soft, dtype=torch.float64),
        torch.as_tensor(hard, dtype=torch.float64),
    )


def _publish_promotion_result(
    output: Path,
    result: PromotionResult,
    eligibility: EligibilityResult,
    hard_manifests: Mapping[int, Path] | None = None,
) -> Path:
    if not result.promoted:
        return save_no_promotion_status(output, result)
    output.mkdir(parents=True, exist_ok=False)
    manifest = output / "manifest.json"
    family_fit_flops = int(
        sum(
            fit["function_evaluations"]
            for audit in result.candidate_audits
            for fit in audit.get("seed_fits", {}).values()
        )
        * 801
        * 8
    )
    nondup_audit_flops = int(
        sum(
            audit.get("duplicate_audit", {}).get("estimated_flops", 0)
            for audit in result.candidate_audits
        )
    )
    _exclusive_json(
        manifest,
        {
            "schema_version": 1,
            "status": "promoted",
            "family_id": result.family_id,
            "primitive_id": result.primitive_id,
            "eligible_seeds": list(eligibility.eligible_seeds),
            "governance": result.governance,
            "candidate_audits": list(result.candidate_audits),
            "accounting": {
                "family_fit_flops": family_fit_flops,
                "nondup_audit_flops": nondup_audit_flops,
                "total_estimated_flops": family_fit_flops + nondup_audit_flops,
            },
            "hard_model_manifests": {
                str(seed): str(path) for seed, path in (hard_manifests or {}).items()
            },
            "publication": {"no_replace": True, "manifest_last": True},
        },
    )
    return manifest


def _selected_checkpoint(model: torch.nn.Module, run: TrainingRun) -> None:
    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("sha256", {}).get("selected_checkpoint")
    if not expected or _sha256(run.checkpoint_path) != expected:
        raise ValueError(f"selected checkpoint hash mismatch: {run.arm}")
    checkpoint = torch.load(run.checkpoint_path, map_location=run.device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    setter = getattr(model, "set_inference_temperature", None)
    if checkpoint.get("inference_temperature") is not None and callable(setter):
        setter(float(checkpoint["inference_temperature"]))


def _neural_accounting(
    run: TrainingRun, parameter_count: int, *, test_rows: int
) -> dict[str, Any]:
    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    accounting = manifest["accounting"]
    return {
        "trainable_parameters": int(parameter_count),
        "steps": int(accounting["optimizer_steps"]),
        "sample_presentations": int(accounting["sample_presentations"]),
        "candidate_evaluations": metric_not_applicable(
            "neural optimizer arm has no candidate-evaluation search"
        ),
        "training_search_flops": int(
            accounting["sample_presentations"]
            * 2
            * accounting["trainable_parameters"]
        ),
        "test_inference_flops": int(test_rows * 2 * parameter_count),
        "wall_clock_seconds": float(accounting["wall_clock_seconds"]),
        "peak_memory_bytes": int(accounting["peak_memory_bytes"]),
    }


def _complexity_metrics(
    complexity: Mapping[str, Any] | None, *, reason: str
) -> dict[str, Any]:
    if complexity is None:
        return {
            key: metric_not_applicable(reason)
            for key in (
                "complexity_ast_nodes",
                "complexity_ast_depth",
                "complexity_free_constants",
                "complexity_serialized_length",
            )
        }
    return {
        "complexity_ast_nodes": int(complexity["ast_node_count"]),
        "complexity_ast_depth": int(complexity["ast_depth"]),
        "complexity_free_constants": int(complexity["free_constants"]),
        "complexity_serialized_length": int(
            complexity["serialized_description_length"]
        ),
    }


def _source_recovery_metrics(
    metadata: Mapping[str, Any] | None, *, reason: str
) -> dict[str, Any]:
    if metadata is None or metadata.get("source") is None:
        not_applicable = metric_not_applicable(reason)
        return {
            "source_recovery_exact": not_applicable,
            "selected_source": not_applicable,
            "selected_window": not_applicable,
            "source_recovery_mass": not_applicable,
        }
    source = str(metadata["source"])
    window = metadata.get("window")
    mass = metadata.get("selected_input_mass")
    return {
        "source_recovery_exact": source == "Return(Close,5)" and window == 5,
        "selected_source": source,
        "selected_window": (
            int(window)
            if window is not None
            else metric_not_applicable("selected source has no fixed lookback window")
        ),
        "source_recovery_mass": (
            float(mass)
            if isinstance(mass, (int, float)) and np.isfinite(float(mass))
            else metric_not_applicable("arm does not expose centered source mass")
        ),
    }


def _edge_source_metadata(
    model: Any, features: torch.Tensor, source_names: Sequence[str] = FEATURE_NAMES
) -> dict[str, Any] | None:
    with torch.no_grad():
        outputs = model.edge_outputs(features)
        centered = outputs - torch.mean(outputs, dim=0, keepdim=True)
        energies = torch.mean(torch.square(centered), dim=0)
        total = torch.sum(energies)
    if not bool(total > 0):
        return None
    masses = energies / total
    index = int(torch.argmax(masses).detach().cpu())
    source = source_names[index]
    window = (
        int(source.removeprefix("Return(Close,").removesuffix(")"))
        if source.startswith("Return(Close,")
        else (20 if "TsMean(Volume,20)" in source else None)
    )
    return {
        "source": source,
        "window": window,
        "selected_input_mass": float(masses[index].detach().cpu()),
    }


def _e5_source_metadata(model: Any) -> dict[str, Any] | None:
    source_mass = model.selected_train_source_mass
    masses = source_mass.get("masses", [])
    if not masses:
        return None
    selected = max(masses, key=lambda value: float(value["mass"]))
    if float(selected["mass"]) <= 0:
        return None
    return {**selected, "selected_input_mass": float(selected["mass"])}


def _complete_arm_metrics(values: Mapping[str, Any], *, reason: str) -> dict[str, Any]:
    result = dict(values)
    for key in PER_ARM_NUMERIC_KEYS:
        result.setdefault(key, metric_not_applicable(f"{reason}: {key}"))
    for key in ("source_recovery_exact", "selected_source", "selected_window"):
        result.setdefault(key, metric_not_applicable(f"{reason}: {key}"))
    return result


def _pretest_seed_summary(
    seed: int,
    replication: GateAReplication,
    root: Path,
    predictors: Mapping[str, Callable[[np.ndarray], np.ndarray]],
    models: Mapping[str, Any],
    runs: Mapping[str, TrainingRun],
    eligibility: EligibilityResult,
    mechanism: Mapping[str, Any],
) -> dict[str, Any]:
    shape_values = {
        arm: evaluate_mechanism_shape(predictor, replication.scaler, mechanism)
        for arm, predictor in predictors.items()
        if arm in {"E1", "E2", "E3", "E4", "E5", "C6", "HARD"}
    }
    shape_path = root / "metrics" / f"seed_{seed}_shape_curves.npz"
    with shape_path.open("xb") as stream:
        np.savez(
            stream,
            mechanism_x=np.linspace(-4.0, 4.0, 801),
            clean_truth=next(iter(shape_values.values()))["clean_truth"],
            **{f"{arm}_prediction": value["prediction"] for arm, value in shape_values.items()},
        )
    e4_model = models["E4"]
    combined = torch.cat((replication.train.features, replication.validation.features))
    test_rows = len(replication.test.features)
    e1_selected = _edge_source_metadata(
        models["E1"], combined.to(run_device(runs["E1"]))
    )
    e2_selected = models["E2"].selected_source_metadata(
        combined.to(run_device(runs["E1"]))
    )
    e3_selected = models["E3"].selected_source_metadata(
        combined.to(run_device(runs["E3"]))
    )
    selected = e4_model.selected_source_metadata(combined.to(run_device(runs["E4"])))
    residual_ratio = float(torch.mean(e4_model.residual_energy(combined.to(run_device(runs["E4"]))).detach().cpu()))
    residual_outputs = e4_model.residual_edge_outputs(
        combined.to(run_device(runs["E4"]))
    )
    residual_raw_energy = float(
        torch.mean(torch.square(residual_outputs)).detach().cpu()
    )
    hard_e3 = models["E3_HARD"]
    hard_e4 = models["E4_HARD_ANALYTICAL"]
    e5 = models["E5"]
    arms: dict[str, Any] = {
        "E1": _complete_arm_metrics({
            "shape_nrmse": shape_values["E1"]["shape_nrmse"],
            **_source_recovery_metrics(
                e1_selected, reason="free-spline contribution energy is zero"
            ),
            **_complexity_metrics(
                None, reason="free spline is not an executable AST"
            ),
            **_neural_accounting(
                runs["E1"], models["E1"].parameter_count, test_rows=test_rows
            ),
            "description_length": metric_not_applicable(
                "free spline is not an executable AST"
            ),
        }, reason="E1 has no independent hard-fidelity target"),
        "E2": _complete_arm_metrics({
            "shape_nrmse": shape_values["E2"]["shape_nrmse"],
            **_source_recovery_metrics(
                e2_selected, reason="post-hoc source contribution energy is zero"
            ),
            "residual_spline_energy_ratio": 0.0,
            "description_length": models["E2"].hard_model.description_length,
            **_complexity_metrics(
                models["E2"].hard_model.complexity(), reason="unreachable"
            ),
            "trainable_parameters": 0,
            "steps": metric_not_applicable("E2 uses bounded post-hoc fits, not optimizer steps"),
            "sample_presentations": metric_not_applicable("E2 has no minibatch optimizer"),
            "candidate_evaluations": models["E2"].fit_count,
            "training_search_flops": int(
                sum(fit.function_evaluations for fit in models["E2"].all_fits)
                * len(replication.train.features)
                * 16
            ),
            "test_inference_flops": metric_not_applicable(
                "prospective test-inference FLOPs are defined only for neural arms"
            ),
            "wall_clock_seconds": models["_e2_wall_clock_seconds"],
            "peak_memory_bytes": metric_not_applicable(
                "E2 post-hoc search has no isolated peak-memory measurement"
            ),
        }, reason="E2 metric is not applicable"),
        "E3": _complete_arm_metrics({
            "shape_nrmse": shape_values["E3"]["shape_nrmse"],
            **_source_recovery_metrics(
                e3_selected, reason="E3 source contribution energy is zero"
            ),
            "residual_spline_energy_ratio": 0.0,
            "description_length": hard_e3.description_length,
            **_complexity_metrics(hard_e3.complexity(), reason="unreachable"),
            **_neural_accounting(
                runs["E3"], models["E3"].parameter_count, test_rows=test_rows
            ),
        }, reason="E3 metric is not applicable"),
        "E4": _complete_arm_metrics({
            "shape_nrmse": shape_values["E4"]["shape_nrmse"],
            **_source_recovery_metrics(
                selected, reason="E4 source contribution energy is zero"
            ),
            "eligible": seed in eligibility.eligible_seeds,
            "residual_spline_energy_ratio": residual_ratio,
            "residual_raw_energy": residual_raw_energy,
            "description_length": hard_e4.description_length,
            **_complexity_metrics(hard_e4.complexity(), reason="unreachable"),
            **_neural_accounting(
                runs["E4"], models["E4"].parameter_count, test_rows=test_rows
            ),
        }, reason="E4 metric is not applicable"),
        "E5": _complete_arm_metrics({
            "shape_nrmse": shape_values["E5"]["shape_nrmse"],
            **_source_recovery_metrics(
                _e5_source_metadata(e5),
                reason="selected E5 expression has zero source contribution energy",
            ),
            "residual_spline_energy_ratio": 0.0,
            "description_length": e5.selected_model.complexity()["serialized_description_length"],
            **_complexity_metrics(e5.selected_model.complexity(), reason="unreachable"),
            "trainable_parameters": 0,
            "steps": metric_not_applicable("E5 is a bounded AST search, not a neural optimizer"),
            "sample_presentations": metric_not_applicable("E5 has no minibatch optimizer"),
            "candidate_evaluations": e5.accounting["distinct_valid_ast_evaluations"],
            "training_search_flops": int(
                sum(
                    entry.get("fit", {}).get("estimated_fit_flops", 0)
                    for entry in e5.ledger
                )
            ),
            "test_inference_flops": metric_not_applicable(
                "prospective test-inference FLOPs are defined only for neural arms"
            ),
            "wall_clock_seconds": e5.accounting["wall_clock_seconds"],
            "peak_memory_bytes": metric_not_applicable(
                "E5 search has no isolated peak-memory measurement"
            ),
        }, reason="E5 metric is not applicable"),
        "C6": _complete_arm_metrics({
            "shape_nrmse": shape_values["C6"]["shape_nrmse"],
            **_source_recovery_metrics(
                None, reason="matched MLP has no additive source selector"
            ),
            **_complexity_metrics(
                None, reason="matched MLP is not an executable AST"
            ),
            "description_length": metric_not_applicable(
                "matched MLP is not an executable AST"
            ),
            **_neural_accounting(
                runs["C6"], models["C6"].parameter_count, test_rows=test_rows
            ),
        }, reason="matched MLP has no independent hard-fidelity target"),
    }
    if "HARD" in predictors:
        hard_model = models["HARD"]
        hard_selected = hard_model.selected_source_metadata(
            combined.to(next(hard_model.buffers()).device)
        )
        arms["HARD"] = _complete_arm_metrics({
            "shape_nrmse": shape_values["HARD"]["shape_nrmse"],
            **_source_recovery_metrics(
                hard_selected, reason="promoted hard source contribution energy is zero"
            ),
            "residual_spline_energy_ratio": 0.0,
            "description_length": hard_model.complexity()["serialized_description_length"],
            **_complexity_metrics(hard_model.complexity(), reason="unreachable"),
            "trainable_parameters": 0,
        }, reason="promoted HARD uses global governance rather than per-seed neural accounting")
    else:
        arms["HARD"] = _complete_arm_metrics(
            {}, reason="no governed primitive was fitted for this seed"
        )
        arms["HARD"]["description_length"] = metric_not_applicable(
            "no governed primitive was fitted for this seed"
        )
    if tuple(arms) != REPORTED_ARMS:
        raise AssertionError("per-seed report arm order differs from registered schema")
    return {"seed": seed, **arms}


def _merge_opened_seed_summary(
    seed: int,
    opening_manifest: Path,
    pretest_summary_path: Path,
) -> dict[str, Any]:
    """Merge only immutable opened predictions and test metrics into pretest evidence."""
    opening = json.loads(opening_manifest.read_text(encoding="utf-8"))
    prediction_path = Path(opening["paths"]["predictions"])
    metrics_path = Path(opening["paths"]["metrics"])
    if _sha256(prediction_path) != opening["sha256"]["predictions"]:
        raise ValueError(f"opened prediction hash mismatch: seed {seed}")
    if _sha256(metrics_path) != opening["sha256"]["metrics"]:
        raise ValueError(f"opened metric hash mismatch: seed {seed}")
    pretest = json.loads(pretest_summary_path.read_text(encoding="utf-8"))
    test_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    with np.load(prediction_path, allow_pickle=False) as arrays:
        arms = {arm: dict(pretest[arm]) for arm in REPORTED_ARMS}
        for arm in REPORTED_ARMS:
            if arm in test_metrics:
                arms[arm].update(
                    clean_nrmse=float(test_metrics[arm]["clean_nrmse"]),
                    noisy_nrmse=float(test_metrics[arm]["noisy_nrmse"]),
                    clean_nrmse_bootstrap=test_metrics[arm]["clean_nrmse_bootstrap"],
                )
        fidelity_pairs = {
            "E2": ("E1_prediction", "E2_prediction"),
            "E3": ("E3_prediction", "E3_HARD_prediction"),
            "E4": ("E4_prediction", "E4_HARD_ANALYTICAL_prediction"),
        }
        if "HARD_prediction" in arrays.files:
            fidelity_pairs["HARD"] = ("E4_prediction", "HARD_prediction")
        for arm, (soft_key, hard_key) in fidelity_pairs.items():
            fidelity = _numpy_fidelity(arrays[soft_key], arrays[hard_key])
            arms[arm].update(
                fidelity_pearson=fidelity["pearson"],
                fidelity_nrmse=fidelity["nrmse"],
                fidelity_max_absolute_error=fidelity["max_absolute_error"],
            )
        if "E5_prediction" in arrays.files:
            arms["E5"].update(
                fidelity_pearson=1.0,
                fidelity_nrmse=0.0,
                fidelity_max_absolute_error=0.0,
            )
    summary = {"seed": int(seed), **arms}
    json.dumps(summary, allow_nan=False)
    return summary


def _terminal_failure_payload(
    *,
    root: Path,
    stage: str,
    error: Exception,
    opened_seeds: Sequence[int],
) -> dict[str, Any]:
    claims = sorted(
        str(path) for path in root.rglob("*.test_once.claim") if path.is_file()
    )
    predictions: dict[str, dict[str, str]] = {}
    for seed in opened_seeds:
        path = root / "predictions" / f"seed_{seed}" / "row_aligned_predictions.npz"
        if path.is_file():
            predictions[str(seed)] = {"path": str(path), "sha256": _sha256(path)}
    return {
        "schema_version": 1,
        "stage": stage,
        "error": {"type": type(error).__name__, "message": str(error)},
        "opened_seeds": [int(seed) for seed in opened_seeds],
        "claims": claims,
        "recovery": {
            "source": "immutable_prediction_artifacts_only",
            "persisted_predictions": predictions,
        },
        "success_manifest_written": False,
        "publication": {"no_replace": True, "terminal": True},
    }


def _artifact_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha256(path)}


def _build_artifact_index(root: Path) -> dict[str, Any]:
    """Index every completed-run regular file without self-referential hashes."""
    matrix_relative = "manifests/matrix.json"
    terminal_relative = "manifests/terminal_failure.json"
    terminal_path = root / terminal_relative
    if terminal_path.exists():
        raise RuntimeError("terminal failure artifact forbids success publication")
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if path.is_symlink():
            raise ValueError(f"artifact tree contains a symbolic link: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in {matrix_relative, terminal_relative}:
            continue
        files[relative] = {
            "path": relative,
            "sha256": _sha256(path),
            "bytes": int(path.stat().st_size),
        }
    aggregate = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "scope": "all_regular_files_below_matrix_root_excluding_top_manifest",
        "files": files,
        "aggregate_sha256": aggregate,
    }


def verify_artifact_index(root: Path | str, manifest_path: Path | str) -> None:
    """Fail closed if any indexed run artifact was added, removed, or changed."""
    root_path = Path(root)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    expected = manifest["artifact_index"]
    actual = _build_artifact_index(root_path)
    if set(actual["files"]) != set(expected["files"]):
        raise ValueError("artifact index file set mismatch")
    for relative, expected_record in expected["files"].items():
        actual_record = actual["files"][relative]
        if actual_record["bytes"] != expected_record["bytes"]:
            raise ValueError(f"artifact index byte-size mismatch: {relative}")
        if actual_record["sha256"] != expected_record["sha256"]:
            raise ValueError(f"artifact index hash mismatch: {relative}")
        if expected_record["path"] != relative:
            raise ValueError(f"artifact index path identity mismatch: {relative}")
    if actual["aggregate_sha256"] != expected["aggregate_sha256"]:
        raise ValueError("artifact index aggregate mismatch")


def _linked_accounting_evidence(
    root: Path,
    settings: MatrixSettings,
    all_runs: Mapping[int, Mapping[str, TrainingRun]],
) -> dict[str, Any]:
    promotion_paths = {
        "promotion": root / "models" / "promotion" / "manifest.json",
        "null_promotion": root / "models" / "null_promotion" / "manifest.json",
    }
    promotion = {}
    for name, path in promotion_paths.items():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        promotion[name] = {
            **_artifact_record(path),
            "accounting": manifest["accounting"],
        }
    controls: dict[str, Any] = {}
    for control, arm in (
        ("null", "E4_NULL"),
        ("source_removed", "E4_SOURCE_REMOVED"),
    ):
        per_seed = {}
        totals = {
            "steps": 0,
            "sample_presentations": 0,
            "training_search_flops": 0,
            "wall_clock_seconds": 0.0,
            "peak_memory_bytes_max": 0,
        }
        for seed in settings.seeds:
            run = all_runs[seed][arm]
            training = json.loads(run.manifest_path.read_text(encoding="utf-8"))[
                "accounting"
            ]
            control_manifest = root / "controls" / control / f"seed_{seed}" / "manifest.json"
            training_flops = int(
                training["sample_presentations"]
                * 2
                * training["trainable_parameters"]
            )
            per_seed[str(seed)] = {
                "control_manifest": _artifact_record(control_manifest),
                "training_manifest": _artifact_record(run.manifest_path),
                "accounting": {
                    "steps": int(training["optimizer_steps"]),
                    "sample_presentations": int(training["sample_presentations"]),
                    "training_search_flops": training_flops,
                    "wall_clock_seconds": float(training["wall_clock_seconds"]),
                    "peak_memory_bytes": int(training["peak_memory_bytes"]),
                },
            }
            totals["steps"] += int(training["optimizer_steps"])
            totals["sample_presentations"] += int(training["sample_presentations"])
            totals["training_search_flops"] += training_flops
            totals["wall_clock_seconds"] += float(training["wall_clock_seconds"])
            totals["peak_memory_bytes_max"] = max(
                totals["peak_memory_bytes_max"], int(training["peak_memory_bytes"])
            )
        controls[control] = {"per_seed": per_seed, "aggregate": totals}
    return {"promotion": promotion, "controls": controls}


def run_device(run: TrainingRun) -> torch.device:
    return torch.device(run.device)


def run_gate_a_matrix(
    settings: MatrixSettings,
    *,
    artifact_base: Path | str = "artifacts",
    device: str = "cuda:0",
    post_open_hook: Callable[[int, Path], None] | None = None,
) -> Path:
    """Run all real arms and controls; test opens only after cross-seed governance."""
    settings.validate()
    if settings.mode == "scientific" and Path(artifact_base) != Path("artifacts"):
        raise ValueError("scientific mode uses fixed project artifacts directory")
    if settings.mode == "scientific" and post_open_hook is not None:
        raise ValueError("scientific matrix forbids test fault-injection hooks")
    seal_before = _seal_snapshot()
    implementation_before = _implementation_snapshot()
    implementation_lock = _verify_implementation_lock()
    root = _prepare_root(settings, artifact_base)
    event_log = root / "ledgers" / "matrix_events.jsonl"
    _append_event(
        event_log,
        "matrix_started",
        mode=settings.mode,
        run_id=settings.run_id,
        seeds=list(settings.seeds),
        device=device,
    )
    replications: dict[int, GateAReplication] = {}
    data_manifests: dict[int, Path] = {}
    all_models: dict[int, dict[str, Any]] = {}
    all_runs: dict[int, dict[str, TrainingRun]] = {}
    residual_shapes: list[ResidualShape] = []
    null_shapes: list[ResidualShape] = []

    for seed in settings.seeds:
        replication = generate_gate_a_replication(settings.config, seed)
        replications[seed] = replication
        data_manifest = save_gate_a_replication(replication, root / "data" / f"seed_{seed}")
        data_manifests[seed] = data_manifest
        models: dict[str, Any] = {
            "E1": FreeSplineKAN(),
            "E3": SymbolicKAN(),
            "E4": SymbolicResidualKAN(),
            "C6": MatchedMLP(initialization_seed=seed),
        }
        runs = {
            arm: _train(
                models[arm],
                replication,
                settings,
                seed=seed,
                arm=arm,
                suffix="main",
                data_manifest=data_manifest,
                root=root,
                device=device,
            )
            for arm in ("E1", "E3", "E4", "C6")
        }
        for arm in runs:
            _selected_checkpoint(models[arm], runs[arm])
        e2_start = time.perf_counter()
        e2 = symbolify_e1(
            models["E1"], replication.train.features.to(run_device(runs["E1"])),
            source_names=FEATURE_NAMES,
        )
        e2_wall_clock_seconds = time.perf_counter() - e2_start
        e2_manifest = save_hard_export(
            e2.hard_model,
            root / "models" / "E2" / f"seed_{seed}",
            arm="E2",
            metadata={"fit_manifest": e2.fit_manifest, "test_once": False},
        )
        e3_hard = models["E3"].harden().cpu()
        e3_hard_manifest = save_hard_export(
            e3_hard,
            root / "models" / "E3_HARD" / f"seed_{seed}",
            arm="E3_HARD",
        )
        e4_hard = models["E4"].harden().cpu()
        e4_hard_manifest = save_hard_export(
            e4_hard,
            root / "models" / "E4_HARD_ANALYTICAL" / f"seed_{seed}",
            arm="E4_HARD_ANALYTICAL",
        )
        e5 = search_e5(
            replication.train.features.numpy(),
            replication.train.noisy_target.to_numpy(),
            replication.validation.features.numpy(),
            replication.validation.clean_truth.to_numpy(),
            settings=E5SearchSettings(
                max_distinct_valid_evaluations=settings.e5_candidate_budget,
                seed=seed,
            ),
        )
        e5_manifest = save_e5_search(
            e5,
            root / "models" / "E5" / f"seed_{seed}",
            metadata={
                "matrix_mode": settings.mode,
                "seed": seed,
                "test_once": False,
            },
        )
        shape = extract_residual_shape(
            models["E4"],
            replication.train.features.to(run_device(runs["E4"])),
            replication.validation.features.to(run_device(runs["E4"])),
            seed=seed,
        )
        save_residual_shape(shape, root / "residual_shapes" / "main" / f"seed_{seed}")
        residual_shapes.append(shape)

        null_train, null_train_evidence = permute_dataset_labels(
            replication.train, block_dates=20, seed=seed + 70_000
        )
        null_validation, null_validation_evidence = permute_dataset_labels(
            replication.validation, block_dates=20, seed=seed + 80_000
        )
        null_manifest = _save_null_control(
            root,
            seed,
            null_train,
            null_validation,
            {"train": null_train_evidence, "validation": null_validation_evidence},
        )
        null_model = SymbolicResidualKAN()
        null_run = _train(
            null_model,
            replication,
            settings,
            seed=seed,
            arm="E4_NULL",
            suffix="null",
            data_manifest=null_manifest,
            root=root,
            device=device,
            train_override=null_train,
            validation_override=null_validation,
        )
        _selected_checkpoint(null_model, null_run)
        null_shape = extract_residual_shape(
            null_model,
            null_train.features.to(run_device(null_run)),
            null_validation.features.to(run_device(null_run)),
            seed=seed,
        )
        null_shape_manifest = save_residual_shape(
            null_shape, root / "residual_shapes" / "null" / f"seed_{seed}"
        )
        _finalize_null_control(
            root,
            seed,
            null_manifest,
            null_shape_manifest,
            null_model,
            null_train,
            null_validation,
            run_device(null_run),
        )
        null_shapes.append(null_shape)

        removed_train, remaining_names = remove_feature_source(
            replication.train, FEATURE_NAMES
        )
        removed_validation, remaining_validation_names = remove_feature_source(
            replication.validation, FEATURE_NAMES
        )
        if remaining_names != remaining_validation_names:
            raise AssertionError("source-removed train/validation identities differ")
        removed_manifest = _save_source_removed_control(
            root, seed, removed_train, removed_validation, remaining_names
        )
        removed_model = SymbolicResidualKAN(
            input_count=len(remaining_names), source_names=remaining_names
        )
        removed_run = _train(
            removed_model,
            replication,
            settings,
            seed=seed,
            arm="E4_SOURCE_REMOVED",
            suffix="source_removed",
            data_manifest=removed_manifest,
            root=root,
            device=device,
            train_override=removed_train,
            validation_override=removed_validation,
        )
        _selected_checkpoint(removed_model, removed_run)
        removed_shape = extract_residual_shape(
            removed_model,
            removed_train.features.to(run_device(removed_run)),
            removed_validation.features.to(run_device(removed_run)),
            seed=seed,
            source_names=remaining_names,
        )
        if removed_shape.source == "Return(Close,5)":
            raise AssertionError("source-removed control reconstructed a forbidden identity")
        removed_shape_manifest = save_residual_shape(
            removed_shape,
            root / "residual_shapes" / "source_removed" / f"seed_{seed}",
        )
        _finalize_source_removed_control(
            root,
            seed,
            removed_manifest,
            removed_shape_manifest,
            removed_model,
            removed_train,
            removed_validation,
            remaining_names,
            run_device(removed_run),
            settings.config["mechanism"],
        )

        models.update(
            {
                "E2": e2,
                "E3_HARD": e3_hard,
                "E4_HARD_ANALYTICAL": e4_hard,
                "E5": e5,
                "_e2_wall_clock_seconds": e2_wall_clock_seconds,
                "E4_NULL": null_model,
                "E4_SOURCE_REMOVED": removed_model,
                "_manifests": {
                    "E2": e2_manifest,
                    "E3_HARD": e3_hard_manifest,
                    "E4_HARD_ANALYTICAL": e4_hard_manifest,
                    "E5": e5_manifest,
                },
            }
        )
        runs.update({"E4_NULL": null_run, "E4_SOURCE_REMOVED": removed_run})
        all_models[seed] = models
        all_runs[seed] = runs
        _append_event(event_log, "seed_selection_frozen", seed=seed)

    eligibility = assess_residual_eligibility(residual_shapes)
    promotion = fit_governed_promotion(residual_shapes, eligibility)
    hard_manifests: dict[int, Path] = {}
    if promotion.promoted:
        assert promotion.family_id is not None
        for seed in eligibility.eligible_seeds:
            replication = replications[seed]
            primitive = PromotedPrimitive.create(
                promotion.family_id, promotion.seed_fits[seed]["parameters"]
            )
            hard_model, evidence = refit_promoted_hard_model(
                all_models[seed]["E4"].harden().cpu(),
                primitive,
                source_index=FEATURE_NAMES.index("Return(Close,5)"),
                train_features=replication.train.features.numpy(),
                train_noisy_target=replication.train.noisy_target.to_numpy(),
                validation_features=replication.validation.features.numpy(),
                validation_clean_truth=replication.validation.clean_truth.to_numpy(),
            )
            hard_manifest = save_promoted_hard_model(
                hard_model,
                root / "models" / "promotion_hard" / f"seed_{seed}",
                evidence=evidence,
            )
            all_models[seed]["HARD"] = hard_model
            all_models[seed]["_manifests"]["HARD"] = hard_manifest
            hard_manifests[seed] = hard_manifest
    _publish_promotion_result(
        root / "models" / "promotion", promotion, eligibility, hard_manifests
    )

    null_eligibility = assess_residual_eligibility(null_shapes)
    null_promotion = fit_governed_promotion(null_shapes, null_eligibility)
    _publish_promotion_result(
        root / "models" / "null_promotion", null_promotion, null_eligibility
    )
    null_promotions = (
        len(null_eligibility.eligible_seeds) if null_promotion.promoted else 0
    )
    _append_event(
        event_log,
        "cross_seed_governance_frozen",
        promotion_status=promotion.status,
        eligible_seeds=list(eligibility.eligible_seeds),
        null_promotion_count=null_promotions,
    )

    all_predictors: dict[int, dict[str, Callable[[np.ndarray], np.ndarray]]] = {}
    all_test_manifests: dict[int, dict[str, Path]] = {}
    pretest_paths: dict[int, Path] = {}
    for seed in settings.seeds:
        models = all_models[seed]
        runs = all_runs[seed]
        predictors: dict[str, Callable[[np.ndarray], np.ndarray]] = {
            "E1": _torch_predictor(models["E1"], device),
            "E2": _torch_predictor(models["E2"].hard_model, device),
            "E3": _torch_predictor(models["E3"], device),
            "E3_HARD": _torch_predictor(models["E3_HARD"], device),
            "E4": _torch_predictor(models["E4"], device),
            "E4_HARD_ANALYTICAL": _torch_predictor(
                models["E4_HARD_ANALYTICAL"], device
            ),
            "E5": lambda matrix, result=models["E5"]: result.selected_model.evaluate(matrix),
            "C6": _torch_predictor(models["C6"], device),
        }
        manifests: dict[str, Path] = {
            "E1": runs["E1"].manifest_path,
            "E2": models["_manifests"]["E2"],
            "E3": runs["E3"].manifest_path,
            "E3_HARD": models["_manifests"]["E3_HARD"],
            "E4": runs["E4"].manifest_path,
            "E4_HARD_ANALYTICAL": models["_manifests"]["E4_HARD_ANALYTICAL"],
            "E5": models["_manifests"]["E5"],
            "C6": runs["C6"].manifest_path,
        }
        if "HARD" in models:
            predictors["HARD"] = _torch_predictor(models["HARD"], device)
            manifests["HARD"] = models["_manifests"]["HARD"]
        all_predictors[seed] = predictors
        all_test_manifests[seed] = manifests
        pretest_summary = _pretest_seed_summary(
            seed,
            replications[seed],
            root,
            predictors,
            models,
            runs,
            eligibility,
            settings.config["mechanism"],
        )
        pretest_path = root / "metrics" / "pretest" / f"seed_{seed}.json"
        _exclusive_json(pretest_path, pretest_summary)
        pretest_paths[seed] = pretest_path

    seal_preopen = _seal_snapshot()
    implementation_preopen = _implementation_snapshot()
    if seal_preopen != seal_before:
        raise RuntimeError("sealed protocol/config changed before test opening")
    if implementation_preopen != implementation_before:
        raise RuntimeError("implementation changed before test opening")
    linked_accounting_path = root / "metrics" / "pretest" / "linked_accounting.json"
    _exclusive_json(
        linked_accounting_path,
        _linked_accounting_evidence(root, settings, all_runs),
    )
    pretest_ready = root / "manifests" / "pretest_ready.json"
    _exclusive_json(
        pretest_ready,
        {
            "schema_version": 1,
            "seed_summaries": {
                str(seed): _artifact_record(path)
                for seed, path in pretest_paths.items()
            },
            "data_manifests": {
                str(seed): _artifact_record(path)
                for seed, path in data_manifests.items()
            },
            "seal": seal_preopen,
            "implementation": implementation_preopen,
            "implementation_lock": implementation_lock,
            "linked_accounting": _artifact_record(linked_accounting_path),
            "test_opened": False,
            "publication": {"no_replace": True, "manifest_last": True},
        },
    )
    _append_event(
        event_log,
        "all_pretest_evidence_frozen",
        artifact_path=str(pretest_ready),
    )

    seed_summaries: dict[int, dict[str, Any]] = {}
    opening_paths: dict[int, Path] = {}
    opened_seeds: list[int] = []
    stage = "before_test_opening"
    try:
        for seed in settings.seeds:
            predictors = all_predictors[seed]
            stage = f"test_open_seed_{seed}"
            opening = open_test_matrix_once(
                replications[seed].test,
                predictors,
                output_directory=root / "predictions" / f"seed_{seed}",
                seed=seed,
                arm_manifest_paths=all_test_manifests[seed],
                bootstrap_replicates=settings.bootstrap_replicates,
                bootstrap_block_dates=20,
            )
            opening_paths[seed] = opening
            opened_seeds.append(seed)
            stage = f"post_open_seed_{seed}"
            if post_open_hook is not None:
                post_open_hook(seed, opening)
            summary = _merge_opened_seed_summary(
                seed, opening, pretest_paths[seed]
            )
            summary_path = root / "metrics" / f"seed_{seed}.json"
            _exclusive_json(summary_path, summary)
            seed_summaries[seed] = summary
            _append_event(
                event_log,
                "unified_test_opening_completed",
                seed=seed,
                arms=sorted(predictors),
            )

        stage = "report_and_gate"
        if _sha256(linked_accounting_path) != json.loads(
            pretest_ready.read_text(encoding="utf-8")
        )["linked_accounting"]["sha256"]:
            raise ValueError("pretest linked-accounting hash mismatch")
        linked_accounting = json.loads(
            linked_accounting_path.read_text(encoding="utf-8")
        )
        if settings.mode == "scientific":
            gate = evaluate_gate_a(seed_summaries, null_promotions=null_promotions)
            gate["linked_accounting"] = linked_accounting
            gate_json, gate_markdown = write_gate_report(
                gate, root / "reports" / "gate_a"
            )
            report_paths = [gate_json, gate_markdown]
        else:
            smoke_json = root / "reports" / "smoke.json"
            _exclusive_json(
                smoke_json,
                {
                    "schema_version": 1,
                    "scientific_evidence": False,
                    "arm_metric_aggregation": aggregate_arm_metrics(seed_summaries),
                    "linked_accounting": linked_accounting,
                    "alpha_profitability_claim": False,
                },
            )
            smoke_report = root / "reports" / "smoke.md"
            smoke_report.write_text(
                "# S1 Gate A matrix smoke\n\n"
                "Fresh-seed reduced execution completed through one-shot test opening.\n\n"
                "This artifact is not scientific evidence and makes no Alpha profitability claim.\n",
                encoding="utf-8",
            )
            report_paths = [smoke_json, smoke_report]

        stage = "final_identity_check"
        seal_after = _seal_snapshot()
        seal_unchanged = seal_before == seal_after
        if not seal_unchanged:
            raise RuntimeError("sealed protocol/config changed during matrix execution")
        implementation_after = _implementation_snapshot()
        implementation_unchanged = implementation_before == implementation_after
        if not implementation_unchanged:
            raise RuntimeError("implementation changed during matrix execution")
        _append_event(
            event_log,
            "matrix_completed",
            seal_unchanged=True,
            implementation_unchanged=True,
        )
        stage = "success_manifest"
        core_children = {
            "data": {
                str(seed): _artifact_record(path)
                for seed, path in data_manifests.items()
            },
            "openings": {
                str(seed): _artifact_record(path)
                for seed, path in opening_paths.items()
            },
            "seed_summaries": {
                str(seed): _artifact_record(root / "metrics" / f"seed_{seed}.json")
                for seed in settings.seeds
            },
            "controls": {
                control: {
                    str(seed): _artifact_record(
                        root / "controls" / control / f"seed_{seed}" / "manifest.json"
                    )
                    for seed in settings.seeds
                }
                for control in ("null", "source_removed")
            },
            "promotion": {
                name: _artifact_record(root / "models" / name / "manifest.json")
                for name in ("promotion", "null_promotion")
            },
            "reports": [_artifact_record(path) for path in report_paths],
            "event_log": _artifact_record(event_log),
            "pretest_ready": _artifact_record(pretest_ready),
        }
        manifest_path = root / "manifests" / "matrix.json"
        artifact_index = _build_artifact_index(root)
        _exclusive_json(
            manifest_path,
            {
            "schema_version": 1,
            "mode": settings.mode,
            "run_id": settings.run_id,
            "seeds": list(settings.seeds),
            "scientific_evidence": settings.mode == "scientific",
            "real_arms": ["E1", "E2", "E3", "E4", "E5", "C6"],
            "controls": ["E4_NULL", "E4_SOURCE_REMOVED"],
            "promotion_status": promotion.status,
            "null_promotion_count": null_promotions,
            "seal_before": seal_before,
            "seal_after": seal_after,
            "seal_unchanged": seal_unchanged,
            "implementation_before": implementation_before,
            "implementation_after": implementation_after,
            "implementation_unchanged": implementation_unchanged,
            "implementation_lock": implementation_lock,
            "reports": [str(path) for path in report_paths],
            "event_log": str(event_log),
            "core_children": core_children,
            "artifact_index": artifact_index,
            "publication": {"no_replace": True, "manifest_last": True},
            },
        )
        return manifest_path
    except Exception as error:
        terminal_path = root / "manifests" / "terminal_failure.json"
        if not terminal_path.exists():
            _exclusive_json(
                terminal_path,
                _terminal_failure_payload(
                    root=root,
                    stage=stage,
                    error=error,
                    opened_seeds=opened_seeds,
                ),
            )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("scientific",), required=True)
    parser.add_argument("--matrix-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    settings = scientific_matrix_settings(args.matrix_id)
    manifest = run_gate_a_matrix(settings, artifact_base="artifacts", device=args.device)
    print(manifest)


if __name__ == "__main__":
    main()
