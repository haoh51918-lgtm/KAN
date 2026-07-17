from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml


def tiny_config() -> dict:
    config = yaml.safe_load(Path("configs/experiments/s1_gate_a_v0.yaml").read_text())
    config = deepcopy(config)
    config["panel"]["assets"] = 1
    config["panel"]["burn_in_dates"] = 20
    config["panel"]["split_dates"] = {"train": 2, "validation": 2, "test": 2}
    return config


def test_generator_matches_worked_first_ohlcv_step_and_draw_order() -> None:
    from mirage_kan.experiments.gate_a import generate_gate_a_replication

    replication = generate_gate_a_replication(tiny_config(), seed=41)
    first = replication.train.panel.raw.iloc[0]
    np.testing.assert_allclose(
        first[["open", "high", "low", "close", "volume"]],
        [
            99.68344696661748,
            100.54393662667299,
            97.8460678276327,
            98.57236632739827,
            959930.9257722675,
        ],
        rtol=1e-14,
    )
    assert replication.provenance["rng"] == "numpy.random.Generator(PCG64)"
    assert replication.provenance["innovation_draw_order"] == [
        "return",
        "volatility",
        "open",
        "high",
        "low",
        "volume",
        "target_noise",
    ]


def test_generator_replays_hash_and_keeps_splits_independent() -> None:
    from mirage_kan.experiments.gate_a import generate_gate_a_replication

    first = generate_gate_a_replication(tiny_config(), seed=41)
    replay = generate_gate_a_replication(tiny_config(), seed=41)
    other = generate_gate_a_replication(tiny_config(), seed=42)

    assert first.provenance["content_sha256"] == replay.provenance["content_sha256"]
    assert first.provenance["content_sha256"] != other.provenance["content_sha256"]
    assert first.train.panel.raw.iloc[0]["close"] != first.validation.panel.raw.iloc[0]["close"]
    assert first.validation.panel.raw.iloc[0]["close"] != first.test.panel.raw.iloc[0]["close"]


def test_candidate_features_use_joint_support_and_train_scaling() -> None:
    from mirage_kan.experiments.gate_a import generate_gate_a_replication

    replication = generate_gate_a_replication(tiny_config(), seed=41)
    assert replication.feature_names == (
        "Return(Close,2)",
        "Return(Close,5)",
        "Return(Close,10)",
        "Return(Close,20)",
        "SafeDiv(Sub(High,Low),Close)",
        "Sub(SafeDiv(Volume,TsMean(Volume,20)),1)",
    )
    assert replication.train.features.shape == (2, 6)
    assert replication.validation.features.shape == (2, 6)
    assert replication.train.index.equals(replication.train.clean_truth.index)
    assert replication.train.support is not replication.train.membership
    assert replication.train.support.all()
    assert replication.train.membership.all()
    np.testing.assert_allclose(
        replication.train.unscaled_features.median(axis=0).to_numpy(),
        replication.scaler.median,
    )
    validation_expected = (
        replication.validation.unscaled_features.to_numpy() - replication.scaler.median
    ) / replication.scaler.iqr
    np.testing.assert_allclose(replication.validation.features.numpy(), validation_expected)
    assert np.isfinite(replication.train.features.numpy()).all()


def test_replication_arrays_are_saved_before_training(tmp_path) -> None:
    from mirage_kan.experiments.gate_a import (
        generate_gate_a_replication,
        save_gate_a_replication,
    )

    replication = generate_gate_a_replication(tiny_config(), seed=41)
    manifest_path = save_gate_a_replication(replication, tmp_path / "generated")
    manifest = yaml.safe_load(manifest_path.read_text())
    assert manifest["content_sha256"] == replication.provenance["content_sha256"]
    assert set(manifest["arrays"]) == {"train", "validation", "test", "scaler"}
    assert all(Path(record["path"]).is_file() for record in manifest["arrays"].values())
    assert all(len(record["sha256"]) == 64 for record in manifest["arrays"].values())

    before = manifest_path.read_bytes()
    with pytest.raises(FileExistsError):
        save_gate_a_replication(replication, tmp_path / "generated")
    assert manifest_path.read_bytes() == before
