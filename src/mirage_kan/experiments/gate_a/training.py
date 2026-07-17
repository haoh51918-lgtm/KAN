"""Shared validation-selected training, checkpoint, accounting, and test-once seam."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .data import GateADataset


@dataclass(frozen=True)
class TrainingSettings:
    """Frozen neural-arm optimizer and budget settings."""

    learning_rate: float = 0.003
    batch_size: int = 2048
    max_steps: int = 2000
    validation_interval_steps: int = 20
    early_stopping_patience_validations: int = 200
    seed: int = 0
    dtype: str = "float64"

    @classmethod
    def from_config(cls, config: dict[str, Any], *, seed: int) -> "TrainingSettings":
        """Construct settings from the sealed training mapping."""
        training = config["training"]
        return cls(
            learning_rate=float(training["learning_rate"]),
            batch_size=int(training["batch_size"]),
            max_steps=int(training["max_steps"]),
            validation_interval_steps=int(training["validation_interval_steps"]),
            early_stopping_patience_validations=int(
                training["early_stopping_patience_validations"]
            ),
            seed=int(seed),
            dtype=str(training["dtype"]),
        )

    def validate(self) -> None:
        """Reject unsupported or non-positive training budgets."""
        if self.dtype != "float64":
            raise ValueError("Gate A neural arms require float64")
        positive = (
            self.learning_rate,
            self.batch_size,
            self.max_steps,
            self.validation_interval_steps,
            self.early_stopping_patience_validations,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("training settings must be positive")


@dataclass(frozen=True)
class TrainingRun:
    """Paths and identity of one selected validation checkpoint."""

    arm: str
    run_id: str
    device: str
    checkpoint_path: Path
    manifest_path: Path
    console_log_path: Path
    test_metrics_path: Path


class _Console:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def write(self, message: str) -> None:
        print(message, flush=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")


def _reserve_run_namespaces(paths: tuple[Path, ...]) -> None:
    """Exclusively reserve every leaf directory used by a run."""
    leaf_directories = tuple(path.parent for path in paths)
    occupied = [path for path in (*leaf_directories, *paths) if path.exists()]
    if occupied:
        raise FileExistsError(f"run namespace already exists: {occupied[0]}")
    for directory in leaf_directories:
        directory.mkdir(parents=True, exist_ok=False)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _exclusive_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _nrmse(prediction: torch.Tensor, truth: torch.Tensor) -> float:
    denominator = torch.std(truth, correction=0)
    if not bool(torch.isfinite(denominator)) or float(denominator) <= 0:
        raise ValueError("NRMSE truth standard deviation must be positive and finite")
    rmse = torch.sqrt(torch.mean((prediction.reshape(-1) - truth.reshape(-1)) ** 2))
    return float((rmse / denominator).detach().cpu())


def _dataset_tensors(
    dataset: GateADataset, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        dataset.features.to(device=device, dtype=torch.float64),
        torch.as_tensor(
            dataset.noisy_target.to_numpy(copy=True),
            dtype=torch.float64,
            device=device,
        ),
        torch.as_tensor(
            dataset.clean_truth.to_numpy(copy=True),
            dtype=torch.float64,
            device=device,
        ),
    )


def _restore_inference_state(model: nn.Module, checkpoint: dict[str, Any]) -> None:
    temperature = checkpoint.get("inference_temperature")
    setter = getattr(model, "set_inference_temperature", None)
    if temperature is not None:
        if not callable(setter):
            raise TypeError("checkpoint has gate temperature but model cannot restore it")
        setter(float(temperature))


def train_and_select(
    model: nn.Module,
    train: GateADataset,
    validation: GateADataset,
    settings: TrainingSettings,
    *,
    arm: str,
    run_id: str,
    data_manifest_path: Path | str,
    artifact_root: Path | str = "artifacts/s1_gate_a",
    log_root: Path | str = "logs",
    device: str | torch.device = "cpu",
) -> TrainingRun:
    """Train any neural arm and select a checkpoint using validation clean NRMSE."""
    settings.validate()
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    artifact_root = Path(artifact_root)
    log_root = Path(log_root)
    checkpoint_path = artifact_root / "checkpoints" / arm / run_id / "selected.pt"
    manifest_path = artifact_root / "manifests" / arm / run_id / "manifest.json"
    test_metrics_path = artifact_root / "metrics" / arm / run_id / "test.json"
    console_log_path = log_root / run_id / "console.log"
    data_manifest_path = Path(data_manifest_path)
    if not data_manifest_path.is_file():
        raise FileNotFoundError("generated data manifest must exist before training")
    _reserve_run_namespaces(
        (checkpoint_path, manifest_path, test_metrics_path, console_log_path)
    )
    console = _Console(console_log_path)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if parameter_count > 10_000:
        raise ValueError("neural arm exceeds the frozen 10,000-parameter ceiling")

    model.to(device=target_device, dtype=torch.float64)
    train_features, train_noisy, _ = _dataset_tensors(train, target_device)
    validation_features, _, validation_clean = _dataset_tensors(
        validation, target_device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(settings.seed)
    sample_presentations = 0
    best_nrmse = float("inf")
    best_step = 0
    validations_without_selection = 0
    steps_completed = 0
    penalty_hook = getattr(model, "training_penalty", None)
    penalty_enabled = callable(penalty_hook)
    penalty_history: list[dict[str, float | int]] = []
    selected_penalty_accounting: dict[str, float] | None = None
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)
    start = time.perf_counter()
    console.write(
        f"run={run_id} arm={arm} device={target_device} dtype=float64 "
        f"parameters={parameter_count} max_steps={settings.max_steps}"
    )

    for step in range(1, settings.max_steps + 1):
        batch_size = min(settings.batch_size, len(train_features))
        positions = torch.randperm(
            len(train_features), generator=generator, device="cpu"
        )[:batch_size].to(target_device)
        optimizer.zero_grad(set_to_none=True)
        batch_features = train_features[positions]
        if penalty_enabled:
            penalty, penalty_accounting = penalty_hook(
                batch_features, step=step, max_steps=settings.max_steps
            )
            penalty_history.append({"step": step, **penalty_accounting})
        else:
            penalty = torch.zeros((), dtype=torch.float64, device=target_device)
            penalty_accounting = None
        prediction = model(batch_features).reshape(-1)
        mse = torch.mean((prediction - train_noisy[positions]) ** 2)
        loss = mse + penalty
        loss.backward()
        optimizer.step()
        steps_completed = step
        sample_presentations += batch_size

        validate_now = (
            step % settings.validation_interval_steps == 0
            or step == settings.max_steps
        )
        if not validate_now:
            continue
        model.eval()
        with torch.no_grad():
            validation_nrmse = _nrmse(
                model(validation_features), validation_clean
            )
        model.train()
        selected = validation_nrmse < best_nrmse - 0.005
        if selected:
            best_nrmse = validation_nrmse
            best_step = step
            validations_without_selection = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint_path.with_suffix(".tmp")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "selected_step": best_step,
                    "validation_clean_nrmse": best_nrmse,
                    "penalty_accounting": penalty_accounting,
                    "inference_temperature": (
                        penalty_accounting.get("temperature")
                        if penalty_accounting is not None
                        else None
                    ),
                },
                temporary,
            )
            os.replace(temporary, checkpoint_path)
            selected_penalty_accounting = penalty_accounting
        else:
            validations_without_selection += 1
        console.write(
            f"step={step} train_mse={float(mse.detach().cpu()):.12g} "
            f"penalty_total={float(penalty.detach().cpu()):.12g} "
            f"validation_clean_nrmse={validation_nrmse:.12g} selected={selected}"
        )
        if (
            validations_without_selection
            >= settings.early_stopping_patience_validations
        ):
            console.write(f"early_stop_step={step}")
            break

    wall_clock_seconds = time.perf_counter() - start
    if best_step == 0:
        raise RuntimeError("no validation checkpoint was selected")
    checkpoint = torch.load(
        checkpoint_path, map_location=target_device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    _restore_inference_state(model, checkpoint)
    if target_device.type == "cuda":
        peak_memory_bytes = int(torch.cuda.max_memory_allocated(target_device))
        peak_memory_kind = "torch_cuda_max_memory_allocated"
    else:
        peak_memory_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        peak_memory_kind = "process_peak_rss_linux"
    manifest = {
        "schema_version": 1,
        "arm": arm,
        "run_id": run_id,
        "model_class": type(model).__name__,
        "model_initialization_seed": getattr(model, "initialization_seed", None),
        "torch_version": torch.__version__,
        "device": str(target_device),
        "settings": asdict(settings),
        "selection": {
            "dataset": "validation",
            "target": "clean_truth",
            "metric": "nrmse",
            "within_0.005_tiebreak": "lower_trainable_parameter_count",
            "selected_step": best_step,
            "selected_value": best_nrmse,
            "selected_inference_temperature": checkpoint.get(
                "inference_temperature"
            ),
        },
        "accounting": {
            "trainable_parameters": parameter_count,
            "optimizer_steps": steps_completed,
            "sample_presentations": sample_presentations,
            "wall_clock_seconds": wall_clock_seconds,
            "peak_memory_bytes": peak_memory_bytes,
            "peak_memory_kind": peak_memory_kind,
        },
        "training_penalties": {
            "enabled": penalty_enabled,
            "model_metadata": (
                model.training_metadata()
                if callable(getattr(model, "training_metadata", None))
                else None
            ),
            "selected_step_accounting": selected_penalty_accounting,
            "step_accounting": penalty_history,
        },
        "paths": {
            "checkpoint": str(checkpoint_path),
            "console_log": str(console_log_path),
            "data_manifest": str(data_manifest_path),
            "manifest": str(manifest_path),
            "test_metrics": str(test_metrics_path),
        },
        "sha256": {
            "selected_checkpoint": hashlib.sha256(
                checkpoint_path.read_bytes()
            ).hexdigest(),
        },
        "test_once": {"evaluated": False},
    }
    console.write(
        f"selected_step={best_step} selected_validation_clean_nrmse={best_nrmse:.12g} "
        f"manifest={manifest_path}"
    )
    _atomic_json(manifest_path, manifest)
    return TrainingRun(
        arm=arm,
        run_id=run_id,
        device=str(target_device),
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        console_log_path=console_log_path,
        test_metrics_path=test_metrics_path,
    )


def evaluate_test_once(
    run: TrainingRun, model: nn.Module, test: GateADataset
) -> dict[str, float]:
    """Read a selected run's test split exactly once and persist both NRMSEs."""
    claim_path = run.test_metrics_path.with_suffix(".claim")
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with claim_path.open("x", encoding="utf-8") as stream:
            stream.write("test read claimed\n")
    except FileExistsError as error:
        raise RuntimeError("test split was already evaluated for this run") from error
    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    if manifest["test_once"]["evaluated"]:
        raise RuntimeError("test split was already evaluated for this run")
    device = torch.device(run.device)
    checkpoint = torch.load(run.checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    _restore_inference_state(model, checkpoint)
    model.to(device=device, dtype=torch.float64)
    features, noisy, clean = _dataset_tensors(test, device)
    model.eval()
    with torch.no_grad():
        prediction = model(features)
        denominator = torch.std(clean, correction=0)
        clean_nrmse = _nrmse(prediction, clean)
        noisy_rmse = torch.sqrt(
            torch.mean((prediction.reshape(-1) - noisy.reshape(-1)) ** 2)
        )
        noisy_nrmse = float((noisy_rmse / denominator).detach().cpu())
    metrics = {"clean_nrmse": clean_nrmse, "noisy_nrmse": noisy_nrmse}
    _exclusive_json(run.test_metrics_path, metrics)
    manifest["test_once"] = {
        "evaluated": True,
        "metrics_path": str(run.test_metrics_path),
    }
    _Console(run.console_log_path).write(
        f"test_once_evaluated=true metrics_path={run.test_metrics_path}"
    )
    _atomic_json(run.manifest_path, manifest)
    return metrics
