"""Evaluator-only Gate A metrics, bootstrap, test opening, and gate decisions."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from .data import GateADataset, TrainScaler

REPORTED_ARMS = ("E1", "E2", "E3", "E4", "E5", "C6", "HARD")
PER_ARM_NUMERIC_KEYS = (
    "clean_nrmse",
    "noisy_nrmse",
    "shape_nrmse",
    "source_recovery_mass",
    "fidelity_pearson",
    "fidelity_nrmse",
    "fidelity_max_absolute_error",
    "residual_spline_energy_ratio",
    "complexity_ast_nodes",
    "complexity_ast_depth",
    "complexity_free_constants",
    "complexity_serialized_length",
    "trainable_parameters",
    "steps",
    "sample_presentations",
    "candidate_evaluations",
    "training_search_flops",
    "test_inference_flops",
    "wall_clock_seconds",
    "peak_memory_bytes",
)


def _finite_vector(values: np.ndarray, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if result.size < 2 or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain at least two finite values")
    return result


def nrmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    prediction = _finite_vector(prediction, "prediction")
    truth = _finite_vector(truth, "truth")
    if prediction.shape != truth.shape:
        raise ValueError("prediction and truth must be row aligned")
    scale = float(np.std(truth, ddof=0))
    if scale <= 0:
        raise ValueError("NRMSE truth variation must be positive")
    return float(np.sqrt(np.mean(np.square(prediction - truth))) / scale)


def clean_mechanism(x: np.ndarray, mechanism: Mapping[str, Any]) -> np.ndarray:
    """Evaluator-only clean mechanism; search and promotion never import it."""
    values = np.asarray(x, dtype=np.float64)
    result = np.empty_like(values)
    negative = values < 0
    result[negative] = float(mechanism["negative_amplitude"]) * (
        1.0 - np.exp(float(mechanism["negative_rate"]) * values[negative])
    )
    result[~negative] = float(mechanism["positive_amplitude"]) * (
        1.0 - np.exp(-float(mechanism["positive_rate"]) * values[~negative])
    )
    return result


def mechanism_shape_inputs(x: np.ndarray, scaler: TrainScaler) -> np.ndarray:
    """Map raw mechanism x to the model's standardized Return5 coordinate."""
    values = np.asarray(x, dtype=np.float64).reshape(-1)
    if scaler.median.shape != (6,) or scaler.iqr.shape != (6,):
        raise ValueError("shape mapping requires the six-source train scaler")
    result = np.zeros((len(values), 6), dtype=np.float64)
    raw_return5 = 0.03 * values
    result[:, 1] = (raw_return5 - scaler.median[1]) / scaler.iqr[1]
    return result


def evaluate_mechanism_shape(
    predictor: Callable[[np.ndarray], np.ndarray],
    scaler: TrainScaler,
    mechanism: Mapping[str, Any],
) -> dict[str, Any]:
    grid = np.linspace(-4.0, 4.0, 801)
    prediction = _finite_vector(predictor(mechanism_shape_inputs(grid, scaler)), "shape prediction")
    truth = clean_mechanism(grid, mechanism)
    return {
        "shape_nrmse": nrmse(prediction, truth),
        "mechanism_grid": grid,
        "prediction": prediction,
        "clean_truth": truth,
        "mapping": "raw_Return5=0.03*x_then_train_median_IQR;others_standardized_zero",
    }


def _date_rectangle(index: pd.MultiIndex) -> tuple[np.ndarray, np.ndarray]:
    dates = index.get_level_values("datetime").unique().to_numpy()
    instruments = index.get_level_values("instrument").unique().to_numpy()
    expected = pd.MultiIndex.from_product(
        [dates, instruments], names=["datetime", "instrument"]
    )
    if not index.equals(expected):
        raise ValueError("bootstrap index must be a complete date-major rectangle")
    return dates, instruments


def date_block_bootstrap_nrmse(
    prediction: np.ndarray,
    truth: np.ndarray,
    index: pd.MultiIndex,
    *,
    seed: int,
    block_dates: int = 20,
    replicates: int = 1000,
) -> dict[str, int | float | str]:
    """Circular moving-date-block percentile interval within one seed."""
    prediction = _finite_vector(prediction, "bootstrap prediction")
    truth = _finite_vector(truth, "bootstrap truth")
    dates, instruments = _date_rectangle(index)
    if prediction.shape != (len(index),) or truth.shape != prediction.shape:
        raise ValueError("bootstrap arrays must align to the provided index")
    if block_dates < 1 or replicates < 1:
        raise ValueError("bootstrap block length and replicate count must be positive")
    assets = len(instruments)
    date_count = len(dates)
    prediction_matrix = prediction.reshape(date_count, assets)
    truth_matrix = truth.reshape(date_count, assets)
    blocks_needed = int(np.ceil(date_count / block_dates))
    rng = np.random.Generator(np.random.PCG64(seed))
    values = np.empty(replicates, dtype=np.float64)
    offsets = np.arange(block_dates)
    for replicate in range(replicates):
        starts = rng.integers(0, date_count, size=blocks_needed)
        sampled_dates = np.concatenate(
            [np.mod(start + offsets, date_count) for start in starts]
        )[:date_count]
        values[replicate] = nrmse(
            prediction_matrix[sampled_dates].reshape(-1),
            truth_matrix[sampled_dates].reshape(-1),
        )
    return {
        "method": "circular_moving_date_block",
        "block_dates": int(block_dates),
        "replicates": int(replicates),
        "seed": int(seed),
        "point": nrmse(prediction, truth),
        "lower": float(np.quantile(values, 0.025)),
        "upper": float(np.quantile(values, 0.975)),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _arm_bootstrap_seed(seed: int, arm: str) -> int:
    arm_offset = int.from_bytes(hashlib.sha256(arm.encode()).digest()[:8], "big")
    return int((int(seed) + 90_000 + arm_offset) % (2**63 - 1))


def open_test_matrix_once(
    test_dataset: GateADataset,
    frozen_predictors: Mapping[str, Callable[[np.ndarray], np.ndarray]],
    *,
    output_directory: Path | str,
    seed: int,
    arm_manifest_paths: Mapping[str, Path | str] | None = None,
    bootstrap_replicates: int = 1000,
    bootstrap_block_dates: int = 20,
) -> Path:
    """Materialize every frozen arm in one exclusive test opening per seed/matrix."""
    output = Path(output_directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_paths = {
        arm: Path(path) for arm, path in (arm_manifest_paths or {}).items()
    }
    if manifest_paths and set(manifest_paths) != set(frozen_predictors):
        raise ValueError("arm manifest set must exactly equal frozen predictor set")
    preflight: dict[str, dict[str, str]] = {}
    for arm, path in manifest_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"arm manifest is missing: {arm}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("test_once", {}).get("evaluated"):
            raise RuntimeError(f"individual arm was already test-evaluated: {arm}")
        preflight[arm] = {"path": str(path), "sha256_before": _sha256(path)}
    matrix_claim_path = output.with_name(output.name + ".test_once.claim")
    if matrix_claim_path.exists() or output.exists():
        raise RuntimeError("seed/matrix test opening was already claimed")
    arm_claims: dict[str, Path] = {}
    for arm in sorted(manifest_paths):
        path = manifest_paths[arm]
        claim_path = path.with_name(path.name + ".test_once.claim")
        try:
            with claim_path.open("x", encoding="utf-8") as stream:
                json.dump(
                    {
                        "schema_version": 1,
                        "arm": arm,
                        "seed": int(seed),
                        "input_manifest": str(path),
                        "input_manifest_sha256": preflight[arm]["sha256_before"],
                        "opening": str(output),
                    },
                    stream,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError as error:
            raise RuntimeError(f"arm test opening was already claimed: {arm}") from error
        arm_claims[arm] = claim_path
    try:
        with matrix_claim_path.open("x", encoding="utf-8") as stream:
            stream.write("exclusive Gate A seed/matrix test opening claimed\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise RuntimeError("seed/matrix test opening was already claimed") from error
    matrix = test_dataset.features.detach().cpu().numpy().copy()
    clean = _finite_vector(test_dataset.clean_truth.to_numpy(), "test clean truth")
    noisy = _finite_vector(test_dataset.noisy_target.to_numpy(), "test noisy target")
    if not frozen_predictors:
        raise ValueError("test opening requires at least one frozen predictor")
    predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for arm in sorted(frozen_predictors):
        prediction = _finite_vector(frozen_predictors[arm](matrix.copy()), f"{arm} prediction")
        if prediction.shape != clean.shape:
            raise ValueError(f"{arm} prediction is not row aligned")
        predictions[arm] = prediction
        clean_nrmse = nrmse(prediction, clean)
        noisy_nrmse = float(
            np.sqrt(np.mean(np.square(prediction - noisy))) / np.std(clean, ddof=0)
        )
        metrics[arm] = {
            "clean_nrmse": clean_nrmse,
            "noisy_nrmse": noisy_nrmse,
            "clean_nrmse_bootstrap": date_block_bootstrap_nrmse(
                prediction,
                clean,
                test_dataset.index,
                seed=_arm_bootstrap_seed(seed, arm),
                block_dates=bootstrap_block_dates,
                replicates=bootstrap_replicates,
            ),
        }
    for arm, path in manifest_paths.items():
        if _sha256(path) != preflight[arm]["sha256_before"]:
            raise RuntimeError(f"input arm manifest changed during test opening: {arm}")
    output.mkdir()
    prediction_path = output / "row_aligned_predictions.npz"
    with prediction_path.open("xb") as stream:
        np.savez(
            stream,
            datetime_ns=test_dataset.index.get_level_values("datetime").asi8,
            instrument=test_dataset.index.get_level_values("instrument").to_numpy(dtype=str),
            clean_truth=clean,
            noisy_target=noisy,
            **{f"{arm}_prediction": value for arm, value in predictions.items()},
        )
        stream.flush()
        os.fsync(stream.fileno())
    metrics_path = output / "metrics.json"
    with metrics_path.open("x", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    receipts_directory = output / "receipts"
    receipts_directory.mkdir()
    receipts: dict[str, str] = {}
    for arm, path in sorted(manifest_paths.items()):
        after = _sha256(path)
        before = preflight[arm]["sha256_before"]
        if after != before:
            raise RuntimeError(f"input arm manifest changed during receipt publication: {arm}")
        receipt_path = receipts_directory / f"{arm}.json"
        with receipt_path.open("x", encoding="utf-8") as stream:
            json.dump(
                {
                    "schema_version": 1,
                    "arm": arm,
                    "seed": int(seed),
                    "input_manifest": str(path),
                    "input_manifest_sha256_before": before,
                    "input_manifest_sha256_after": after,
                    "claim": str(arm_claims[arm]),
                    "claim_sha256": _sha256(arm_claims[arm]),
                    "opening": str(output),
                    "prediction_key": f"{arm}_prediction",
                    "metrics": metrics[arm],
                    "publication": {"no_replace": True},
                },
                stream,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        receipts[arm] = str(receipt_path)
    manifest_path = output / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "schema_version": 1,
                "seed": int(seed),
                "arms": sorted(predictions),
                "single_opening": True,
                "paths": {
                    "predictions": str(prediction_path),
                    "metrics": str(metrics_path),
                    "claim": str(matrix_claim_path),
                },
                "sha256": {
                    "predictions": _sha256(prediction_path),
                    "metrics": _sha256(metrics_path),
                },
                "preflight": {"arm_manifests": preflight},
                "arm_claims": {arm: str(path) for arm, path in arm_claims.items()},
                "receipts": receipts,
                "publication": {"no_replace": True, "manifest_last": True},
            },
            stream,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return manifest_path


def metric_not_applicable(reason: str) -> dict[str, str]:
    if not reason:
        raise ValueError("N/A metric requires an explicit reason")
    return {"value": "N/A", "reason": reason}


def _numeric_value(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def aggregate_arm_metrics(
    seed_metrics: Mapping[int, Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Aggregate every registered numeric metric with explicit missingness."""
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in REPORTED_ARMS:
        arm_result: dict[str, dict[str, Any]] = {}
        for key in PER_ARM_NUMERIC_KEYS:
            available: list[tuple[int, float]] = []
            missing: list[int] = []
            for seed, metrics in seed_metrics.items():
                numeric = _numeric_value(metrics.get(arm, {}).get(key))
                if numeric is None:
                    missing.append(int(seed))
                else:
                    available.append((int(seed), numeric))
            reason = f"{arm}.{key} has no finite value on any seed"
            arm_result[key] = {
                "median": (
                    float(np.median([value for _, value in available]))
                    if available
                    else metric_not_applicable(reason)
                ),
                "full_seed_range": (
                    [
                        min(value for _, value in available),
                        max(value for _, value in available),
                    ]
                    if available
                    else metric_not_applicable(reason)
                ),
                "available_seeds": [seed for seed, _ in available],
                "missing_seeds": missing,
            }
        result[arm] = arm_result
    return result


def _median(seed_metrics: Mapping[int, Mapping[str, Any]], arm: str, key: str) -> float:
    return float(np.median([float(value[arm][key]) for value in seed_metrics.values()]))


def _median_seeds(
    seed_metrics: Mapping[int, Mapping[str, Any]],
    seeds: list[int],
    arm: str,
    key: str,
) -> float:
    values = []
    for seed in seeds:
        value = seed_metrics[seed].get(arm, {}).get(key)
        if isinstance(value, (int, float)) and np.isfinite(value):
            values.append(float(value))
    if len(values) != len(seeds) or not values:
        return float("nan")
    return float(np.median(values))


def _metric_range(
    seed_metrics: Mapping[int, Mapping[str, Any]], arm: str, key: str
) -> dict[str, Any]:
    available = []
    missing = []
    for seed, metrics in seed_metrics.items():
        value = metrics.get(arm, {}).get(key)
        if isinstance(value, (int, float)) and np.isfinite(value):
            available.append((seed, float(value)))
        else:
            missing.append(seed)
    return {
        "value": (
            [min(value for _, value in available), max(value for _, value in available)]
            if available
            else metric_not_applicable(f"{arm}.{key} has no finite seed value")
        ),
        "available_seeds": [seed for seed, _ in available],
        "missing_seeds": missing,
        "missing_reason": (
            "arm_or_metric_not_applicable_for_non-successful_seed" if missing else None
        ),
    }


def _condition(passed: bool, evidence: dict[str, Any]) -> dict[str, Any]:
    return {"passed": bool(passed), "evidence": evidence}


def _finite_or_na(value: float, reason: str) -> float | dict[str, str]:
    return float(value) if np.isfinite(value) else metric_not_applicable(reason)


def evaluate_gate_a(
    seed_metrics: Mapping[int, Mapping[str, Any]], *, null_promotions: int
) -> dict[str, Any]:
    """Apply all seven sealed Gate A decisions literally to three-seed metrics."""
    if len(seed_metrics) != 3:
        raise ValueError("Gate A aggregation requires exactly three seeds")
    e1 = _median(seed_metrics, "E1", "clean_nrmse")
    c6 = _median(seed_metrics, "C6", "clean_nrmse")
    capacity = min(e1, c6) <= 0.15
    successful = [
        seed
        for seed, metrics in seed_metrics.items()
        if metrics["E4"]["selected_source"] == "Return(Close,5)"
        and bool(metrics["E4"]["eligible"])
    ]
    stable = len(successful) >= 2
    e4_shape = _median(seed_metrics, "E4", "shape_nrmse")
    better_symbolic_shape = min(
        _median(seed_metrics, "E3", "shape_nrmse"),
        _median(seed_metrics, "E5", "shape_nrmse"),
    )
    relative_improvement = (better_symbolic_shape - e4_shape) / better_symbolic_shape
    successful_shapes = [
        float(seed_metrics[seed]["E4"]["shape_nrmse"]) for seed in successful
    ]
    shape_quality = bool(
        successful
        and e4_shape <= 0.12
        and max(successful_shapes) <= 0.18
        and relative_improvement >= 0.15
    )
    e4_clean = _median(seed_metrics, "E4", "clean_nrmse")
    numerical = e4_clean <= 1.15 * min(e1, c6)
    hard_clean = _median_seeds(seed_metrics, successful, "HARD", "clean_nrmse")
    hard_fidelity_nrmse = _median_seeds(
        seed_metrics, successful, "HARD", "fidelity_nrmse"
    )
    hard_correlations = []
    for seed in successful:
        value = seed_metrics[seed].get("HARD", {}).get("fidelity_pearson", "N/A")
        if isinstance(value, (int, float)) and np.isfinite(value):
            hard_correlations.append(float(value))
    executable = bool(
        successful
        and np.isfinite(hard_clean)
        and np.isfinite(hard_fidelity_nrmse)
        and len(hard_correlations) == len(successful)
        and hard_clean <= 0.15
        and min(hard_correlations) >= 0.98
        and hard_fidelity_nrmse <= 0.10
    )
    hard_axes = (
        hard_clean,
        hard_fidelity_nrmse,
        _median_seeds(
            seed_metrics, successful, "HARD", "complexity_serialized_length"
        ),
    )
    dominators = []
    for arm in ("E2", "E3", "E5"):
        axes = (
            _median_seeds(seed_metrics, successful, arm, "clean_nrmse"),
            _median_seeds(seed_metrics, successful, arm, "fidelity_nrmse"),
            _median_seeds(
                seed_metrics, successful, arm, "complexity_serialized_length"
            ),
        )
        if all(np.isfinite(value) for value in axes + hard_axes) and all(
            left <= right for left, right in zip(axes, hard_axes, strict=True)
        ) and any(
            left < right for left, right in zip(axes, hard_axes, strict=True)
        ):
            dominators.append(arm)
    pareto = bool(all(np.isfinite(value) for value in hard_axes) and not dominators)
    null_safe = int(null_promotions) == 0
    conditions = {
        "1": _condition(capacity, {"E1_median": e1, "C6_median": c6}),
        "2": _condition(stable, {"successful_seeds": successful}),
        "3": _condition(
            shape_quality,
            {
                "E4_median": e4_shape,
                "successful_seed_values": successful_shapes,
                "better_E3_E5_median": better_symbolic_shape,
                "relative_improvement": relative_improvement,
            },
        ),
        "4": _condition(
            numerical,
            {"E4_median": e4_clean, "best_numeric_median": min(e1, c6)},
        ),
        "5": _condition(
            executable,
            {
                "hard_median_clean_nrmse": _finite_or_na(
                    hard_clean, "no governed hard model on every successful seed"
                ),
                "successful_seed_correlations": hard_correlations,
                "hard_median_fidelity_nrmse": _finite_or_na(
                    hard_fidelity_nrmse, "no governed hard model on every successful seed"
                ),
            },
        ),
        "6": _condition(
            pareto,
            {
                "hard_axes": [
                    _finite_or_na(
                        value, "no governed hard model on every successful seed"
                    )
                    for value in hard_axes
                ],
                "dominators": dominators,
            },
        ),
        "7": _condition(null_safe, {"null_promotions": int(null_promotions)}),
    }
    if not capacity:
        status = "capacity_inconclusive"
    elif all(condition["passed"] for condition in conditions.values()):
        status = "pass"
    else:
        status = "scientific_fail"
    return {
        "schema_version": 1,
        "status": status,
        "conditions": conditions,
        "aggregation": "median_and_full_seed_range",
        "seed_ranges": {
            arm: {
                key: _metric_range(seed_metrics, arm, key)
                for key in (
                    "clean_nrmse",
                    "shape_nrmse",
                    "fidelity_nrmse",
                    "complexity_serialized_length",
                )
            }
            for arm in ("E1", "E2", "E3", "E4", "E5", "C6", "HARD")
        },
        "arm_metric_aggregation": aggregate_arm_metrics(seed_metrics),
        "population_p_value": "not_computed_from_three_seeds",
        "alpha_profitability_claim": False,
    }


def write_gate_report(result: Mapping[str, Any], output_directory: Path | str) -> tuple[Path, Path]:
    """Publish machine JSON and a compact human-readable non-profitability report."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    json_path = output / "gate_a.json"
    with json_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    markdown_path = output / "gate_a.md"
    rows = ["# S1 Gate A", "", f"Status: **{result['status']}**", "", "| Gate | Result |", "|---:|:---:|"]
    rows.extend(
        f"| {index} | {'PASS' if condition['passed'] else 'FAIL'} |"
        for index, condition in result["conditions"].items()
    )
    rows.extend(
        [
            "",
            "This synthetic mechanism gate does not establish Alpha profitability.",
            "No population p-value is claimed from three seeds.",
        ]
    )
    markdown_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return json_path, markdown_path
