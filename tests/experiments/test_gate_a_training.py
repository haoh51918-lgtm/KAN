from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml


def test_shared_training_selects_on_validation_and_accounts_budget(
    tmp_path, monkeypatch
) -> None:
    import mirage_kan.experiments.gate_a.training as training_module

    from mirage_kan.experiments.gate_a.models import FreeSplineKAN
    from mirage_kan.experiments.gate_a.training import (
        TrainingSettings,
        evaluate_test_once,
        train_and_select,
    )
    from mirage_kan.experiments.gate_a import generate_gate_a_replication
    from mirage_kan.experiments.gate_a import save_gate_a_replication

    publication_events = []
    original_atomic_json = training_module._atomic_json
    original_exclusive_json = training_module._exclusive_json
    original_console_write = training_module._Console.write

    def record_manifest(path, value):
        original_atomic_json(path, value)
        publication_events.append("manifest")

    def record_metric(path, value):
        original_exclusive_json(path, value)
        publication_events.append("metric")

    def record_log(console, message):
        original_console_write(console, message)
        publication_events.append("log")

    monkeypatch.setattr(training_module, "_atomic_json", record_manifest)
    monkeypatch.setattr(training_module, "_exclusive_json", record_metric)
    monkeypatch.setattr(training_module._Console, "write", record_log)

    config = deepcopy(
        yaml.safe_load(Path("configs/experiments/s1_gate_a_v0.yaml").read_text())
    )
    config["panel"]["assets"] = 1
    config["panel"]["burn_in_dates"] = 20
    config["panel"]["split_dates"] = {"train": 2, "validation": 2, "test": 2}
    replication = generate_gate_a_replication(config, seed=41)
    data_manifest = save_gate_a_replication(
        replication, tmp_path / "artifacts" / "data" / "tiny_e1"
    )
    settings = TrainingSettings(
        learning_rate=0.003,
        batch_size=2,
        max_steps=3,
        validation_interval_steps=1,
        early_stopping_patience_validations=3,
        seed=7,
    )
    model = FreeSplineKAN()
    run = train_and_select(
        model,
        replication.train,
        replication.validation,
        settings,
        arm="E1",
        run_id="tiny_e1",
        data_manifest_path=data_manifest,
        artifact_root=tmp_path / "artifacts",
        log_root=tmp_path / "logs",
    )
    manifest = json.loads(run.manifest_path.read_text())
    assert run.manifest_path.stat().st_mtime_ns >= run.console_log_path.stat().st_mtime_ns
    assert run.manifest_path.stat().st_mtime_ns >= run.checkpoint_path.stat().st_mtime_ns
    assert run.checkpoint_path.is_file()
    assert run.console_log_path.is_file()
    assert manifest["selection"]["dataset"] == "validation"
    assert manifest["selection"]["target"] == "clean_truth"
    assert manifest["test_once"]["evaluated"] is False
    assert manifest["accounting"]["trainable_parameters"] == 115
    assert manifest["accounting"]["optimizer_steps"] == 3
    assert manifest["accounting"]["sample_presentations"] == 6
    assert manifest["accounting"]["wall_clock_seconds"] >= 0
    assert manifest["accounting"]["peak_memory_bytes"] >= 0
    assert manifest["paths"]["console_log"] == str(run.console_log_path)
    assert manifest["paths"]["data_manifest"] == str(data_manifest)
    assert manifest["sha256"]["selected_checkpoint"] == hashlib.sha256(
        run.checkpoint_path.read_bytes()
    ).hexdigest()
    assert publication_events[-1] == "manifest"

    publication_events.clear()
    metrics = evaluate_test_once(run, model, replication.test)
    assert set(metrics) == {"clean_nrmse", "noisy_nrmse"}
    assert run.test_metrics_path.is_file()
    assert run.manifest_path.stat().st_mtime_ns >= run.console_log_path.stat().st_mtime_ns
    assert run.manifest_path.stat().st_mtime_ns >= run.test_metrics_path.stat().st_mtime_ns
    assert publication_events[-3:] == ["metric", "log", "manifest"]
    with pytest.raises(RuntimeError, match="already evaluated"):
        evaluate_test_once(run, model, replication.test)


def test_shared_trainer_applies_and_persists_symbolic_penalties(tmp_path) -> None:
    from mirage_kan.experiments.gate_a import (
        generate_gate_a_replication,
        save_gate_a_replication,
    )
    from mirage_kan.experiments.gate_a.models import SymbolicKAN
    from mirage_kan.experiments.gate_a.training import (
        TrainingSettings,
        train_and_select,
    )

    config = deepcopy(
        yaml.safe_load(Path("configs/experiments/s1_gate_a_v0.yaml").read_text())
    )
    config["panel"]["assets"] = 1
    config["panel"]["burn_in_dates"] = 20
    config["panel"]["split_dates"] = {"train": 2, "validation": 2, "test": 2}
    replication = generate_gate_a_replication(config, seed=41)
    data_manifest = save_gate_a_replication(replication, tmp_path / "data")
    run = train_and_select(
        SymbolicKAN(),
        replication.train,
        replication.validation,
        TrainingSettings(
            batch_size=2,
            max_steps=2,
            validation_interval_steps=1,
            early_stopping_patience_validations=2,
        ),
        arm="E3",
        run_id="penalty_literal",
        data_manifest_path=data_manifest,
        artifact_root=tmp_path / "artifacts",
        log_root=tmp_path / "logs",
    )
    manifest = json.loads(run.manifest_path.read_text())
    penalties = manifest["training_penalties"]
    assert penalties["enabled"] is True
    assert len(penalties["step_accounting"]) == 2
    assert penalties["step_accounting"][0]["temperature"] == pytest.approx(1.05)
    assert penalties["step_accounting"][1]["temperature"] == pytest.approx(0.1)
    assert penalties["selected_step_accounting"]["total"] > 0
    checkpoint = torch.load(run.checkpoint_path, weights_only=True)
    assert checkpoint["penalty_accounting"]["temperature"] == pytest.approx(0.1)
    assert "penalty_total=" in run.console_log_path.read_text()


def test_training_reserves_every_run_namespace_without_replacement(tmp_path) -> None:
    from mirage_kan.experiments.gate_a import (
        generate_gate_a_replication,
        save_gate_a_replication,
    )
    from mirage_kan.experiments.gate_a.models import FreeSplineKAN
    from mirage_kan.experiments.gate_a.training import TrainingSettings, train_and_select

    config = deepcopy(
        yaml.safe_load(Path("configs/experiments/s1_gate_a_v0.yaml").read_text())
    )
    config["panel"]["assets"] = 1
    config["panel"]["burn_in_dates"] = 20
    config["panel"]["split_dates"] = {"train": 2, "validation": 2, "test": 2}
    replication = generate_gate_a_replication(config, seed=41)
    data_manifest = save_gate_a_replication(replication, tmp_path / "data")
    occupied_log = tmp_path / "logs" / "occupied" / "console.log"
    occupied_log.parent.mkdir(parents=True)
    occupied_log.write_text("do not replace\n")
    with pytest.raises(FileExistsError, match="run namespace"):
        train_and_select(
            FreeSplineKAN(),
            replication.train,
            replication.validation,
            TrainingSettings(max_steps=1, validation_interval_steps=1),
            arm="E1",
            run_id="occupied",
            data_manifest_path=data_manifest,
            artifact_root=tmp_path / "artifacts",
            log_root=tmp_path / "logs",
        )
    assert occupied_log.read_text() == "do not replace\n"


def test_selected_symbolic_temperature_is_restored_when_best_step_is_early(
    tmp_path,
) -> None:
    from mirage_kan.experiments.gate_a import (
        generate_gate_a_replication,
        save_gate_a_replication,
    )
    from mirage_kan.experiments.gate_a.models import SymbolicKAN
    from mirage_kan.experiments.gate_a.training import TrainingSettings, train_and_select

    config = deepcopy(
        yaml.safe_load(Path("configs/experiments/s1_gate_a_v0.yaml").read_text())
    )
    config["panel"]["assets"] = 1
    config["panel"]["burn_in_dates"] = 20
    config["panel"]["split_dates"] = {"train": 2, "validation": 2, "test": 2}
    replication = generate_gate_a_replication(config, seed=41)
    data_manifest = save_gate_a_replication(replication, tmp_path / "data")
    model = SymbolicKAN()
    run = train_and_select(
        model,
        replication.train,
        replication.validation,
        TrainingSettings(
            learning_rate=1e-12,
            batch_size=2,
            max_steps=2,
            validation_interval_steps=1,
            early_stopping_patience_validations=2,
        ),
        arm="E3",
        run_id="early_temperature",
        data_manifest_path=data_manifest,
        artifact_root=tmp_path / "artifacts",
        log_root=tmp_path / "logs",
    )
    manifest = json.loads(run.manifest_path.read_text())
    assert manifest["selection"]["selected_step"] == 1
    assert manifest["selection"]["selected_inference_temperature"] == pytest.approx(
        1.05
    )
    assert model._training_temperature == pytest.approx(1.05)
    checkpoint = torch.load(run.checkpoint_path, weights_only=True)
    assert checkpoint["inference_temperature"] == pytest.approx(1.05)
