"""Exact synthetic OHLCV panels for the sealed S1 Gate A protocol."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import yaml

from mirage_kan.data import PitPanel
from mirage_kan.data.pit import sha256_file
from mirage_kan.dsl import AstNode
from mirage_kan.executor import evaluate_torch

FEATURE_NAMES = (
    "Return(Close,2)",
    "Return(Close,5)",
    "Return(Close,10)",
    "Return(Close,20)",
    "SafeDiv(Sub(High,Low),Close)",
    "Sub(SafeDiv(Volume,TsMean(Volume,20)),1)",
)


@dataclass(frozen=True)
class TrainScaler:
    """Train-only robust feature location and scale."""

    median: np.ndarray
    iqr: np.ndarray


@dataclass(frozen=True)
class GateADataset:
    """One independent split with typed features and mechanism truth."""

    panel: PitPanel
    index: pd.MultiIndex
    unscaled_features: pd.DataFrame
    features: torch.Tensor
    clean_truth: pd.Series
    noisy_target: pd.Series
    membership: pd.Series
    support: pd.Series


@dataclass(frozen=True)
class GateAReplication:
    """Train/validation/test splits sharing only train-fitted scaling."""

    train: GateADataset
    validation: GateADataset
    test: GateADataset
    feature_names: tuple[str, ...]
    scaler: TrainScaler
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class _RawSplit:
    panel: PitPanel
    retained_index: pd.MultiIndex
    clean_truth: pd.Series
    target_noise: np.ndarray


def _load_config(
    config: Mapping[str, Any] | str | Path,
) -> tuple[dict[str, Any], str | None]:
    if isinstance(config, Mapping):
        return dict(config), None
    path = Path(config)
    with path.open(encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise ValueError("Gate A config must be a mapping")
    return loaded, sha256_file(path)


def _candidate_programs() -> tuple[AstNode, ...]:
    close = AstNode("Close")
    volume = AstNode("Volume")
    returns = tuple(
        AstNode("Return", (close,), {"window": window})
        for window in (2, 5, 10, 20)
    )
    range_close = AstNode(
        "SafeDiv",
        (AstNode("Sub", (AstNode("High"), AstNode("Low"))), close),
    )
    relative_volume = AstNode(
        "Sub",
        (
            AstNode(
                "SafeDiv",
                (volume, AstNode("TsMean", (volume,), {"window": 20})),
            ),
            AstNode("Constant", params={"value": 1.0}),
        ),
    )
    return (*returns, range_close, relative_volume)


def _generate_raw_split(
    config: Mapping[str, Any], seed: int, split: str
) -> _RawSplit:
    panel_config = config["panel"]
    burn_in = int(panel_config["burn_in_dates"])
    split_dates = int(panel_config["split_dates"][split])
    dates = burn_in + split_dates
    assets = int(panel_config["assets"])
    offset = int(panel_config["split_seed_offsets"][split])
    rng = np.random.Generator(np.random.PCG64(seed + offset))
    innovations = {
        name: rng.standard_normal((dates, assets))
        for name in panel_config["innovation_draw_order"]
    }

    close = np.empty((dates, assets), dtype=np.float64)
    open_ = np.empty_like(close)
    high = np.empty_like(close)
    low = np.empty_like(close)
    volume = np.empty_like(close)
    previous_close = np.full(assets, float(panel_config["initial_close"]))
    previous_return = np.full(assets, float(panel_config["initial_return"]))
    h = np.full(assets, float(panel_config["initial_log_volatility"]))
    u = np.full(assets, float(panel_config["initial_log_volume"]))
    return_config = panel_config["return_process"]
    volatility_config = panel_config["log_volatility_process"]
    volume_config = panel_config["log_volume_process"]

    for date in range(dates):
        h = (
            float(volatility_config["ar"]) * h
            + float(volatility_config["innovation_scale"])
            * innovations["volatility"][date]
        )
        current_return = (
            float(return_config["ar"]) * previous_return
            + float(return_config["base_scale"])
            * np.exp(h / 2)
            * innovations["return"][date]
        )
        open_[date] = previous_close * np.exp(
            float(panel_config["overnight_scale"]) * innovations["open"][date]
        )
        close[date] = previous_close * np.exp(current_return)
        high[date] = np.maximum(open_[date], close[date]) * np.exp(
            float(panel_config["intraday_spread_scale"])
            * np.abs(innovations["high"][date])
        )
        low[date] = np.minimum(open_[date], close[date]) * np.exp(
            -float(panel_config["intraday_spread_scale"])
            * np.abs(innovations["low"][date])
        )
        u = (
            float(volume_config["ar"]) * u
            + float(volume_config["abs_return_loading"]) * np.abs(current_return)
            + float(volume_config["innovation_scale"])
            * innovations["volume"][date]
        )
        volume[date] = float(panel_config["volume_level"]) * np.exp(u)
        previous_close = close[date]
        previous_return = current_return

    datetimes = pd.date_range("2000-01-01", periods=dates, freq="D")
    instruments = [f"asset_{asset:03d}" for asset in range(assets)]
    index = pd.MultiIndex.from_product(
        [datetimes, instruments], names=["datetime", "instrument"]
    )
    frame = pd.DataFrame(
        {
            "datetime": index.get_level_values("datetime"),
            "instrument": index.get_level_values("instrument"),
            "open": open_.reshape(-1),
            "high": high.reshape(-1),
            "low": low.reshape(-1),
            "close": close.reshape(-1),
            "volume": volume.reshape(-1),
            "in_universe": True,
        }
    )
    panel = PitPanel.from_frame(frame)
    retained_index = index[burn_in * assets :]
    driver = evaluate_torch(
        AstNode("Return", (AstNode("Close"),), {"window": 5}), panel
    ).values.loc[retained_index]
    mechanism = config["mechanism"]
    x = np.clip(
        driver.to_numpy() / float(mechanism["input_scale"]),
        *map(float, mechanism["input_clip"]),
    )
    clean = np.where(
        x < 0,
        float(mechanism["negative_amplitude"])
        * (1 - np.exp(float(mechanism["negative_rate"]) * x)),
        float(mechanism["positive_amplitude"])
        * (1 - np.exp(-float(mechanism["positive_rate"]) * x)),
    )
    return _RawSplit(
        panel=panel,
        retained_index=retained_index,
        clean_truth=pd.Series(clean, index=retained_index, name="clean_truth"),
        target_noise=innovations["target_noise"][burn_in:].reshape(-1),
    )


def _features(raw: _RawSplit) -> tuple[pd.DataFrame, pd.Series]:
    evaluations = [evaluate_torch(program, raw.panel) for program in _candidate_programs()]
    support = pd.concat(
        [result.support.loc[raw.retained_index] for result in evaluations], axis=1
    ).all(axis=1)
    values = pd.concat(
        [result.values.loc[raw.retained_index] for result in evaluations], axis=1
    )
    values.columns = FEATURE_NAMES
    support = support.rename("joint_expression_support")
    if not bool(support.all()):
        values = values.loc[support]
    return values, support.loc[values.index]


def _dataset(
    raw: _RawSplit,
    values: pd.DataFrame,
    support: pd.Series,
    scaler: TrainScaler,
    noise_scale: float,
) -> GateADataset:
    scaled = (values.to_numpy() - scaler.median) / scaler.iqr
    clean = raw.clean_truth.loc[values.index]
    noise_positions = raw.retained_index.get_indexer(values.index)
    noisy = clean.to_numpy() + noise_scale * raw.target_noise[noise_positions]
    membership = raw.panel.membership.loc[values.index].copy().rename("membership")
    return GateADataset(
        panel=raw.panel,
        index=values.index,
        unscaled_features=values,
        features=torch.as_tensor(scaled, dtype=torch.float64),
        clean_truth=clean,
        noisy_target=pd.Series(noisy, index=values.index, name="noisy_target"),
        membership=membership,
        support=support.copy(),
    )


def _content_hash(datasets: tuple[GateADataset, ...]) -> str:
    digest = hashlib.sha256()
    for dataset in datasets:
        for array in (
            dataset.panel.raw.to_numpy(dtype=np.float64),
            dataset.features.numpy(),
            dataset.clean_truth.to_numpy(),
            dataset.noisy_target.to_numpy(),
        ):
            digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def generate_gate_a_replication(
    config: Mapping[str, Any] | str | Path, seed: int
) -> GateAReplication:
    """Generate one exact independent Gate A replication from a sealed config."""
    loaded, source_config_sha256 = _load_config(config)
    if tuple(loaded["candidate_inputs"]) != FEATURE_NAMES:
        raise ValueError("candidate inputs differ from the frozen Gate A feature set")
    raw = {
        split: _generate_raw_split(loaded, seed, split)
        for split in ("train", "validation", "test")
    }
    feature_data = {split: _features(value) for split, value in raw.items()}
    train_values = feature_data["train"][0]
    median = train_values.median(axis=0).to_numpy(dtype=np.float64)
    q75 = train_values.quantile(0.75, axis=0).to_numpy(dtype=np.float64)
    q25 = train_values.quantile(0.25, axis=0).to_numpy(dtype=np.float64)
    iqr = q75 - q25
    if not np.isfinite(median).all() or not np.isfinite(iqr).all() or (iqr <= 0).any():
        raise ValueError("train feature median/IQR is non-finite or degenerate")
    scaler = TrainScaler(median=median, iqr=iqr)
    noise_scale = (
        float(loaded["mechanism"]["noise_fraction_of_train_clean_std"])
        * raw["train"].clean_truth.loc[train_values.index].to_numpy().std(ddof=0)
    )
    datasets = {
        split: _dataset(
            raw[split], *feature_data[split], scaler=scaler, noise_scale=noise_scale
        )
        for split in ("train", "validation", "test")
    }
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "seed": int(seed),
        "numpy_version": np.__version__,
        "rng": "numpy.random.Generator(PCG64)",
        "innovation_draw_order": list(loaded["panel"]["innovation_draw_order"]),
        "split_seed_offsets": dict(loaded["panel"]["split_seed_offsets"]),
        "candidate_ast_identities": [program.identity for program in _candidate_programs()],
        "resolved_config_sha256": hashlib.sha256(
            json.dumps(loaded, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    if source_config_sha256 is not None:
        provenance["source_config_sha256"] = source_config_sha256
    ordered = (datasets["train"], datasets["validation"], datasets["test"])
    provenance["content_sha256"] = _content_hash(ordered)
    return GateAReplication(
        train=datasets["train"],
        validation=datasets["validation"],
        test=datasets["test"],
        feature_names=FEATURE_NAMES,
        scaler=scaler,
        provenance=provenance,
    )


def save_gate_a_replication(
    replication: GateAReplication, output_directory: Path | str
) -> Path:
    """Persist every returned panel/feature/target array and a hash manifest."""
    output = Path(output_directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    arrays_directory = output / "arrays"
    arrays_directory.mkdir()
    records: dict[str, dict[str, str]] = {}
    for split in ("train", "validation", "test"):
        dataset = getattr(replication, split)
        path = arrays_directory / f"{split}.npz"
        panel_index = dataset.panel.raw.index
        np.savez(
            path,
            panel_datetime=panel_index.get_level_values("datetime").asi8,
            panel_instrument=panel_index.get_level_values("instrument").to_numpy(
                dtype=str
            ),
            panel_ohlcv=dataset.panel.raw.to_numpy(dtype=np.float64),
            retained_datetime=dataset.index.get_level_values("datetime").asi8,
            retained_instrument=dataset.index.get_level_values(
                "instrument"
            ).to_numpy(dtype=str),
            unscaled_features=dataset.unscaled_features.to_numpy(dtype=np.float64),
            scaled_features=dataset.features.numpy(),
            clean_truth=dataset.clean_truth.to_numpy(),
            noisy_target=dataset.noisy_target.to_numpy(),
            membership=dataset.membership.to_numpy(dtype=bool),
            joint_expression_support=dataset.support.to_numpy(dtype=bool),
        )
        records[split] = {"path": str(path), "sha256": sha256_file(path)}
    scaler_path = arrays_directory / "scaler.npz"
    np.savez(
        scaler_path,
        median=replication.scaler.median,
        iqr=replication.scaler.iqr,
        feature_names=np.asarray(replication.feature_names, dtype=str),
    )
    records["scaler"] = {
        "path": str(scaler_path),
        "sha256": sha256_file(scaler_path),
    }
    manifest_path = output / "manifest.json"
    manifest = {
        "schema_version": 1,
        "content_sha256": replication.provenance["content_sha256"],
        "provenance": dict(replication.provenance),
        "feature_names": list(replication.feature_names),
        "arrays": records,
    }
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(manifest_path)
    return manifest_path
