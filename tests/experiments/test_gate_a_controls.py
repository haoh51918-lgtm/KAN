from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch


def _index(dates: int = 6, assets: int = 3) -> pd.MultiIndex:
    return pd.MultiIndex.from_product(
        [pd.date_range("2026-01-01", periods=dates), [f"a{i}" for i in range(assets)]],
        names=["datetime", "instrument"],
    )


def test_date_block_permutation_moves_whole_cross_sections_without_row_reorder() -> None:
    from mirage_kan.experiments.gate_a.controls import date_block_permute

    index = _index()
    values = np.arange(len(index), dtype=np.float64)
    permuted, evidence = date_block_permute(values, index, block_dates=2, seed=91)
    assert evidence["nonzero_circular_date_shift"] in {2, 4}
    assert evidence["feature_row_order_changed"] is False
    matrix = values.reshape(6, 3)
    expected = np.roll(matrix, -evidence["nonzero_circular_date_shift"], axis=0)
    np.testing.assert_array_equal(permuted.reshape(6, 3), expected)
    for date in range(6):
        # Every destination date receives one intact source-date cross-section.
        assert np.diff(permuted.reshape(6, 3)[date]).tolist() == [1.0, 1.0]
    with pytest.raises(ValueError, match="strictly more dates"):
        date_block_permute(values[:6], _index(2), block_dates=2, seed=1)


def test_null_control_permutes_train_and_selection_targets_and_source_removal_is_exact() -> None:
    from mirage_kan.experiments.gate_a.controls import (
        permute_dataset_labels,
        remove_feature_source,
    )
    from mirage_kan.experiments.gate_a.data import FEATURE_NAMES, GateADataset

    index = _index(dates=4, assets=2)
    values = np.arange(len(index), dtype=np.float64)
    dataset = GateADataset(
        panel=None,  # Controls operate only on the already-materialized matrix seam.
        index=index,
        unscaled_features=pd.DataFrame(
            np.arange(len(index) * 6).reshape(len(index), 6),
            index=index,
            columns=FEATURE_NAMES,
        ),
        features=torch.arange(len(index) * 6, dtype=torch.float64).reshape(len(index), 6),
        clean_truth=pd.Series(values, index=index),
        noisy_target=pd.Series(values + 0.25, index=index),
        membership=pd.Series(True, index=index),
        support=pd.Series(True, index=index),
    )
    null, evidence = permute_dataset_labels(dataset, block_dates=1, seed=17)
    assert evidence["training_objective_permuted"] is True
    assert evidence["validation_selection_permuted"] is True
    assert not np.array_equal(null.clean_truth.to_numpy(), values)
    np.testing.assert_array_equal(null.features.numpy(), dataset.features.numpy())

    removed, names = remove_feature_source(dataset, FEATURE_NAMES, "Return(Close,5)")
    assert "Return(Close,5)" not in names
    assert names == FEATURE_NAMES[:1] + FEATURE_NAMES[2:]
    assert removed.features.shape[1] == 5
    np.testing.assert_array_equal(
        removed.features.numpy(), np.delete(dataset.features.numpy(), 1, axis=1)
    )
    assert list(removed.unscaled_features.columns) == list(names)

