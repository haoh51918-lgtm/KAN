from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch


def test_shape_coordinate_mapping_matches_worked_literal_and_target_is_evaluator_only() -> None:
    from mirage_kan.experiments.gate_a.data import TrainScaler
    from mirage_kan.experiments.gate_a.metrics import (
        clean_mechanism,
        mechanism_shape_inputs,
    )

    scaler = TrainScaler(
        median=np.array([9.0, 0.01, 8.0, 7.0, 6.0, 5.0]),
        iqr=np.array([2.0, 0.02, 2.0, 2.0, 2.0, 2.0]),
    )
    x = np.array([-4.0, 0.0, 4.0])
    mechanism = {
        "negative_amplitude": -0.6,
        "negative_rate": 2.5,
        "positive_amplitude": 1.8,
        "positive_rate": 0.25,
    }
    inputs = mechanism_shape_inputs(x, scaler)
    np.testing.assert_allclose(inputs[:, 1], [-6.5, -0.5, 5.5])
    np.testing.assert_array_equal(np.delete(inputs, 1, axis=1), np.zeros((3, 5)))
    np.testing.assert_allclose(
        clean_mechanism(x, mechanism),
        [
            -0.6 * (1 - np.exp(-10.0)),
            0.0,
            1.8 * (1 - np.exp(-1.0)),
        ],
    )


def test_date_block_bootstrap_is_deterministic_and_reports_percentile_interval() -> None:
    from mirage_kan.experiments.gate_a.metrics import date_block_bootstrap_nrmse

    index = pd.MultiIndex.from_product(
        [pd.date_range("2026-01-01", periods=30), ["a", "b"]],
        names=["datetime", "instrument"],
    )
    truth = np.linspace(-2, 2, len(index))
    prediction = truth + np.sin(np.arange(len(index))) * 0.1
    first = date_block_bootstrap_nrmse(
        prediction, truth, index, seed=551, block_dates=5, replicates=100
    )
    second = date_block_bootstrap_nrmse(
        prediction, truth, index, seed=551, block_dates=5, replicates=100
    )
    assert first == second
    assert first["method"] == "circular_moving_date_block"
    assert first["replicates"] == 100
    assert first["lower"] <= first["point"] <= first["upper"]
    assert "p_value" not in first


def test_unified_test_opening_is_exclusive_row_aligned_and_manifest_last(tmp_path) -> None:
    from mirage_kan.experiments.gate_a.data import FEATURE_NAMES, GateADataset
    from mirage_kan.experiments.gate_a.metrics import open_test_matrix_once

    index = pd.MultiIndex.from_product(
        [pd.date_range("2026-01-01", periods=25), ["a", "b"]],
        names=["datetime", "instrument"],
    )
    features = np.arange(len(index) * 6, dtype=np.float64).reshape(len(index), 6) / 100
    clean = 2.0 * features[:, 0] - features[:, 1]
    dataset = GateADataset(
        panel=None,
        index=index,
        unscaled_features=pd.DataFrame(features, index=index, columns=FEATURE_NAMES),
        features=torch.as_tensor(features),
        clean_truth=pd.Series(clean, index=index),
        noisy_target=pd.Series(clean + 0.1, index=index),
        membership=pd.Series(True, index=index),
        support=pd.Series(True, index=index),
    )
    manifests = {}
    original_bytes = {}
    for arm in ("literal", "zero"):
        path = tmp_path / f"{arm}.json"
        path.write_text(json.dumps({"test_once": {"evaluated": False}}))
        manifests[arm] = path
        original_bytes[arm] = path.read_bytes()
    manifest_path = open_test_matrix_once(
        dataset,
        {
            "literal": lambda matrix: 2.0 * matrix[:, 0] - matrix[:, 1],
            "zero": lambda matrix: np.zeros(len(matrix)),
        },
        output_directory=tmp_path / "opening",
        seed=123,
        arm_manifest_paths=manifests,
        bootstrap_replicates=20,
        bootstrap_block_dates=5,
    )
    manifest = json.loads(manifest_path.read_text())
    arrays = np.load(manifest["paths"]["predictions"], allow_pickle=False)
    np.testing.assert_array_equal(arrays["literal_prediction"], clean)
    np.testing.assert_array_equal(arrays["clean_truth"], clean)
    assert arrays["datetime_ns"].shape == clean.shape
    assert manifest["arms"] == ["literal", "zero"]
    assert manifest["publication"]["manifest_last"] is True
    for arm, input_manifest in manifests.items():
        assert input_manifest.read_bytes() == original_bytes[arm]
        preflight = manifest["preflight"]["arm_manifests"][arm]
        assert preflight["path"] == str(input_manifest)
        assert preflight["sha256_before"] == hashlib.sha256(
            original_bytes[arm]
        ).hexdigest()
        claim = input_manifest.with_name(input_manifest.name + ".test_once.claim")
        assert claim.is_file()
        receipt = json.loads(
            (manifest_path.parent / "receipts" / f"{arm}.json").read_text()
        )
        assert receipt["input_manifest_sha256_before"] == preflight["sha256_before"]
        assert receipt["input_manifest_sha256_after"] == preflight["sha256_before"]
        assert receipt["prediction_key"] == f"{arm}_prediction"
    with pytest.raises(RuntimeError, match="already claimed"):
        open_test_matrix_once(
            dataset,
            {"literal": lambda matrix: clean},
            output_directory=tmp_path / "opening",
            seed=123,
        )


def test_failed_unified_opening_keeps_claim_but_publishes_no_manifest_or_arm_update(tmp_path) -> None:
    from mirage_kan.experiments.gate_a.data import FEATURE_NAMES, GateADataset
    from mirage_kan.experiments.gate_a.metrics import open_test_matrix_once

    index = pd.MultiIndex.from_product(
        [pd.date_range("2026-01-01", periods=4), ["a", "b"]],
        names=["datetime", "instrument"],
    )
    features = np.ones((len(index), 6))
    truth = np.arange(len(index), dtype=np.float64)
    dataset = GateADataset(
        panel=None,
        index=index,
        unscaled_features=pd.DataFrame(features, index=index, columns=FEATURE_NAMES),
        features=torch.as_tensor(features),
        clean_truth=pd.Series(truth, index=index),
        noisy_target=pd.Series(truth, index=index),
        membership=pd.Series(True, index=index),
        support=pd.Series(True, index=index),
    )
    arm_manifest = tmp_path / "arm.json"
    arm_manifest.write_text(json.dumps({"test_once": {"evaluated": False}}))
    before = arm_manifest.read_bytes()

    def broken(_: np.ndarray) -> np.ndarray:
        raise RuntimeError("intentional predictor failure")

    output = tmp_path / "failed_opening"
    with pytest.raises(RuntimeError, match="intentional predictor failure"):
        open_test_matrix_once(
            dataset,
            {"broken": broken},
            output_directory=output,
            seed=5,
            arm_manifest_paths={"broken": arm_manifest},
        )
    assert output.with_name(output.name + ".test_once.claim").is_file()
    assert not (output / "manifest.json").exists()
    assert arm_manifest.read_bytes() == before
    assert arm_manifest.with_name(arm_manifest.name + ".test_once.claim").is_file()
    assert not (output / "receipts").exists()


def test_unified_opening_preflights_all_manifests_before_any_update(tmp_path) -> None:
    from mirage_kan.experiments.gate_a.data import FEATURE_NAMES, GateADataset
    from mirage_kan.experiments.gate_a.metrics import open_test_matrix_once

    index = pd.MultiIndex.from_product(
        [pd.date_range("2026-01-01", periods=2), ["a", "b"]],
        names=["datetime", "instrument"],
    )
    features = np.ones((4, 6))
    dataset = GateADataset(
        panel=None,
        index=index,
        unscaled_features=pd.DataFrame(features, index=index, columns=FEATURE_NAMES),
        features=torch.as_tensor(features),
        clean_truth=pd.Series([0.0, 1.0, 2.0, 3.0], index=index),
        noisy_target=pd.Series([0.0, 1.0, 2.0, 3.0], index=index),
        membership=pd.Series(True, index=index),
        support=pd.Series(True, index=index),
    )
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    third = tmp_path / "third.json"
    first.write_text(json.dumps({"test_once": {"evaluated": False}}))
    second.write_text(json.dumps({"test_once": {"evaluated": False}}))
    third.write_text(json.dumps({"test_once": {"evaluated": True}}))
    predictors = {arm: (lambda matrix: np.arange(len(matrix))) for arm in ("a", "b", "c")}
    with pytest.raises(RuntimeError, match="already test-evaluated"):
        open_test_matrix_once(
            dataset,
            predictors,
            output_directory=tmp_path / "preflight",
            seed=4,
            arm_manifest_paths={"a": first, "b": second, "c": third},
        )
    assert json.loads(first.read_text())["test_once"]["evaluated"] is False
    assert not (tmp_path / "preflight.test_once.claim").exists()


def test_partial_arm_claim_collision_fails_closed_without_opening(tmp_path) -> None:
    from mirage_kan.experiments.gate_a.data import FEATURE_NAMES, GateADataset
    from mirage_kan.experiments.gate_a.metrics import open_test_matrix_once

    index = pd.MultiIndex.from_product(
        [pd.date_range("2026-01-01", periods=2), ["a", "b"]],
        names=["datetime", "instrument"],
    )
    features = np.ones((4, 6))
    dataset = GateADataset(
        panel=None,
        index=index,
        unscaled_features=pd.DataFrame(features, index=index, columns=FEATURE_NAMES),
        features=torch.as_tensor(features),
        clean_truth=pd.Series([0.0, 1.0, 2.0, 3.0], index=index),
        noisy_target=pd.Series([0.0, 1.0, 2.0, 3.0], index=index),
        membership=pd.Series(True, index=index),
        support=pd.Series(True, index=index),
    )
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text(json.dumps({"test_once": {"evaluated": False}}))
    second.write_text(json.dumps({"test_once": {"evaluated": False}}))
    second_claim = second.with_name(second.name + ".test_once.claim")
    second_claim.write_text("already claimed\n")
    output = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="arm test opening was already claimed"):
        open_test_matrix_once(
            dataset,
            {"a": lambda matrix: np.arange(len(matrix)), "b": lambda matrix: np.arange(len(matrix))},
            output_directory=output,
            seed=7,
            arm_manifest_paths={"a": first, "b": second},
        )
    assert first.with_name(first.name + ".test_once.claim").is_file()
    assert second_claim.read_text() == "already claimed\n"
    assert not output.exists()
    assert not output.with_name(output.name + ".test_once.claim").exists()


def _passing_gate_fixture() -> dict[int, dict[str, object]]:
    values = {}
    for seed, delta in zip((1729, 2718, 31415), (0.0, 0.002, -0.002), strict=True):
        values[seed] = {
            "E1": {"clean_nrmse": 0.10 + delta},
            "E2": {"clean_nrmse": 0.14 + delta, "fidelity_nrmse": 0.12, "complexity_serialized_length": 500},
            "E3": {"clean_nrmse": 0.13 + delta, "shape_nrmse": 0.16, "fidelity_nrmse": 0.11, "complexity_serialized_length": 460},
            "E4": {
                "clean_nrmse": 0.11 + delta,
                "shape_nrmse": 0.10,
                "selected_source": "Return(Close,5)",
                "eligible": True,
            },
            "E5": {"clean_nrmse": 0.15 + delta, "shape_nrmse": 0.14, "fidelity_nrmse": 0.0, "complexity_serialized_length": 600},
            "C6": {"clean_nrmse": 0.105 + delta},
            "HARD": {
                "clean_nrmse": 0.12 + delta,
                "fidelity_pearson": 0.995,
                "fidelity_nrmse": 0.05,
                "complexity_serialized_length": 420,
            },
        }
    return values


def test_all_seven_gate_decisions_and_capacity_inconclusive_are_literal() -> None:
    from mirage_kan.experiments.gate_a.metrics import (
        PER_ARM_NUMERIC_KEYS,
        REPORTED_ARMS,
        evaluate_gate_a,
    )

    passing = _passing_gate_fixture()
    result = evaluate_gate_a(passing, null_promotions=0)
    assert result["status"] == "pass"
    assert all(result["conditions"][str(index)]["passed"] for index in range(1, 8))
    for arm in REPORTED_ARMS:
        assert set(result["arm_metric_aggregation"][arm]) == set(
            PER_ARM_NUMERIC_KEYS
        )
        for aggregate in result["arm_metric_aggregation"][arm].values():
            assert set(aggregate) == {
                "median",
                "full_seed_range",
                "available_seeds",
                "missing_seeds",
            }
            assert aggregate["available_seeds"] or aggregate["missing_seeds"]
    clean_aggregate = result["arm_metric_aggregation"]["E1"]["clean_nrmse"]
    assert clean_aggregate["median"] == pytest.approx(0.10)
    assert clean_aggregate["full_seed_range"] == pytest.approx([0.098, 0.102])

    two_successful = _passing_gate_fixture()
    two_successful[31415]["E4"]["eligible"] = False
    del two_successful[31415]["HARD"]
    subset_result = evaluate_gate_a(two_successful, null_promotions=0)
    assert subset_result["status"] == "pass"
    assert subset_result["conditions"]["5"]["passed"] is True
    assert subset_result["seed_ranges"]["HARD"]["clean_nrmse"] == {
        "value": [0.12, 0.122],
        "available_seeds": [1729, 2718],
        "missing_seeds": [31415],
        "missing_reason": "arm_or_metric_not_applicable_for_non-successful_seed",
    }

    no_promotion = _passing_gate_fixture()
    for metrics in no_promotion.values():
        del metrics["HARD"]
    no_promotion_result = evaluate_gate_a(no_promotion, null_promotions=0)
    assert no_promotion_result["status"] == "scientific_fail"
    assert no_promotion_result["conditions"]["5"]["passed"] is False
    assert no_promotion_result["conditions"]["6"]["passed"] is False
    serialized = json.dumps(no_promotion_result, allow_nan=False)
    assert "NaN" not in serialized and "Infinity" not in serialized

    for condition in range(1, 8):
        fixture = _passing_gate_fixture()
        null_promotions = 0
        if condition == 1:
            for metrics in fixture.values():
                metrics["E1"]["clean_nrmse"] = 0.3
                metrics["C6"]["clean_nrmse"] = 0.3
        elif condition == 2:
            for metrics in fixture.values():
                metrics["E4"]["eligible"] = False
        elif condition == 3:
            for metrics in fixture.values():
                metrics["E4"]["shape_nrmse"] = 0.2
        elif condition == 4:
            for metrics in fixture.values():
                metrics["E4"]["clean_nrmse"] = 0.2
        elif condition == 5:
            for metrics in fixture.values():
                metrics["HARD"]["fidelity_pearson"] = 0.9
        elif condition == 6:
            for metrics in fixture.values():
                metrics["E3"].update(
                    clean_nrmse=0.10,
                    fidelity_nrmse=0.04,
                    complexity_serialized_length=400,
                )
        else:
            null_promotions = 1
        failed = evaluate_gate_a(fixture, null_promotions=null_promotions)
        assert failed["conditions"][str(condition)]["passed"] is False
        if condition == 1:
            assert failed["status"] == "capacity_inconclusive"
        else:
            assert failed["status"] == "scientific_fail"
