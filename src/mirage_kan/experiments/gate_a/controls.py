"""Leakage-resistant negative controls for the S1 Gate A matrix."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np
import pandas as pd

from .data import GateADataset


def _date_major_shape(index: pd.MultiIndex) -> tuple[int, int]:
    if index.names[:2] != ["datetime", "instrument"]:
        raise ValueError("control index must be datetime/instrument date-major")
    dates = index.get_level_values("datetime").unique()
    instruments = index.get_level_values("instrument").unique()
    expected = pd.MultiIndex.from_product(
        [dates, instruments], names=["datetime", "instrument"]
    )
    if not index.equals(expected):
        raise ValueError("control index must be a complete date-major rectangle")
    return len(dates), len(instruments)


def date_block_permute(
    values: np.ndarray | Sequence[float],
    index: pd.MultiIndex,
    *,
    block_dates: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, int | bool | str]]:
    """Circularly shift labels by a nonzero whole-date block offset.

    This is a bijection of label dates. Feature rows are never reordered, and every
    asset in one destination date receives labels from the same source date.
    """
    if block_dates < 1:
        raise ValueError("date block length must be positive")
    dates, assets = _date_major_shape(index)
    if dates < 2 * block_dates:
        raise ValueError("block permutation needs strictly more dates than one block")
    array = np.asarray(values)
    if array.shape != (len(index),):
        raise ValueError("control labels must have one scalar per index row")
    if not np.isfinite(array).all():
        raise ValueError("control labels must be finite")
    whole_block_shifts = dates // block_dates - 1
    block_offset = int(
        np.random.Generator(np.random.PCG64(seed)).integers(
            1, whole_block_shifts + 1
        )
    )
    date_shift = block_offset * block_dates
    result = np.roll(array.reshape(dates, assets), -date_shift, axis=0).reshape(-1)
    return result, {
        "method": "nonzero_circular_whole_date_block_shift",
        "seed": int(seed),
        "block_dates": int(block_dates),
        "nonzero_circular_date_shift": int(date_shift),
        "feature_row_order_changed": False,
        "whole_cross_section_preserved": True,
        "row_or_future_feature_access": False,
    }


def permute_dataset_labels(
    dataset: GateADataset, *, block_dates: int, seed: int
) -> tuple[GateADataset, dict[str, object]]:
    """Destroy train and validation relationships with one shared label mapping."""
    clean, clean_evidence = date_block_permute(
        dataset.clean_truth.to_numpy(),
        dataset.index,
        block_dates=block_dates,
        seed=seed,
    )
    noisy, noisy_evidence = date_block_permute(
        dataset.noisy_target.to_numpy(),
        dataset.index,
        block_dates=block_dates,
        seed=seed,
    )
    if clean_evidence["nonzero_circular_date_shift"] != noisy_evidence[
        "nonzero_circular_date_shift"
    ]:
        raise AssertionError("clean and noisy null targets must share one mapping")
    controlled = replace(
        dataset,
        clean_truth=pd.Series(clean, index=dataset.index, name="null_clean_truth"),
        noisy_target=pd.Series(noisy, index=dataset.index, name="null_noisy_target"),
    )
    return controlled, {
        **clean_evidence,
        "training_objective_permuted": True,
        "validation_selection_permuted": True,
        "feature_order_preserved": True,
        "interpretation": (
            "same whole-date label permutation is used for noisy training objective "
            "and clean validation selection"
        ),
    }


def remove_feature_source(
    dataset: GateADataset,
    source_names: Sequence[str],
    source: str = "Return(Close,5)",
) -> tuple[GateADataset, tuple[str, ...]]:
    """Delete exactly one materialized source column without reconstruction."""
    names = tuple(source_names)
    if names.count(source) != 1 or dataset.features.shape[1] != len(names):
        raise ValueError("source removal requires one exact source and aligned columns")
    position = names.index(source)
    keep = [index for index in range(len(names)) if index != position]
    remaining = tuple(names[index] for index in keep)
    frame = dataset.unscaled_features.iloc[:, keep].copy()
    frame.columns = remaining
    controlled = replace(
        dataset,
        unscaled_features=frame,
        features=dataset.features[:, keep].clone(),
    )
    return controlled, remaining

