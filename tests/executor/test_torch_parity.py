from __future__ import annotations

import numpy as np
import pandas as pd

from mirage_kan.data import PitPanel
from mirage_kan.dsl import AstNode, evaluate


def test_torch_leaf_preserves_index_and_observed_support() -> None:
    from mirage_kan.executor import evaluate_torch

    panel = PitPanel.from_frame(
        pd.DataFrame(
            {
                "datetime": pd.to_datetime(
                    ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"]
                ),
                "instrument": ["A", "B", "A", "B"],
                "open": [10.0, 20.0, 11.0, 21.0],
                "high": [10.0, 20.0, 11.0, 21.0],
                "low": [10.0, 20.0, 11.0, 21.0],
                "close": [10.0, 20.0, np.nan, 21.0],
                "volume": [100.0, 200.0, 110.0, 210.0],
                "in_universe": [False, True, True, True],
            }
        )
    )

    result = evaluate_torch(AstNode("Close"), panel, device="cpu")
    expected_values = pd.Series(
        [10.0, 20.0, np.nan, 21.0], index=panel.raw.index, name="close"
    )
    expected_support = pd.Series(
        [True, True, False, True], index=panel.raw.index, name="support"
    )
    pd.testing.assert_series_equal(result.values, expected_values)
    pd.testing.assert_series_equal(result.support, expected_support)
    assert bool(result.support.iloc[0]) and not bool(panel.membership.iloc[0])
    assert not bool(result.support.iloc[2]) and bool(panel.membership.iloc[2])


def test_torch_binary_operators_match_worked_results(tiny_panel: PitPanel) -> None:
    from mirage_kan.executor import evaluate_torch

    programs_and_expected = (
        (
            AstNode("Add", (AstNode("Close"), AstNode("Open"))),
            [19.5, 39.5, 21.5, 41.5, np.nan, 43.5, 25.5, 45.5, 27.5, 47.5],
        ),
        (
            AstNode(
                "SafeDiv",
                (
                    AstNode("Sub", (AstNode("High"), AstNode("Low"))),
                    AstNode("Close"),
                ),
            ),
            [0.2, 0.1, 2 / 11, 2 / 21, np.nan, 2 / 22, 2 / 13, 2 / 23, 2 / 14, 2 / 24],
        ),
    )

    for program, expected in programs_and_expected:
        result = evaluate_torch(program, tiny_panel, device="cpu")
        reference = evaluate(program, tiny_panel)
        np.testing.assert_allclose(result.values, expected, equal_nan=True)
        pd.testing.assert_series_equal(result.values, reference.values)
        pd.testing.assert_series_equal(result.support, reference.support)


def test_torch_lag_operators_are_causal_at_window_boundaries(
    tiny_panel: PitPanel,
) -> None:
    from mirage_kan.executor import evaluate_torch

    programs_and_expected = (
        (
            AstNode("Delay", (AstNode("Close"),), {"window": 2}),
            [np.nan, np.nan, np.nan, np.nan, 10.0, 20.0, 11.0, 21.0, np.nan, 22.0],
        ),
        (
            AstNode("Delta", (AstNode("Close"),), {"window": 2}),
            [np.nan, np.nan, np.nan, np.nan, np.nan, 2.0, 2.0, 2.0, np.nan, 2.0],
        ),
        (
            AstNode("Return", (AstNode("Close"),), {"window": 2}),
            [np.nan, np.nan, np.nan, np.nan, np.nan, 0.1, 2 / 11, 2 / 21, np.nan, 1 / 11],
        ),
    )

    for program, expected in programs_and_expected:
        result = evaluate_torch(program, tiny_panel, device="cpu")
        reference = evaluate(program, tiny_panel)
        np.testing.assert_allclose(result.values, expected, equal_nan=True)
        pd.testing.assert_series_equal(result.values, reference.values)
        pd.testing.assert_series_equal(result.support, reference.support)


def test_torch_rolling_mean_requires_complete_causal_windows(
    tiny_panel: PitPanel,
) -> None:
    from mirage_kan.executor import evaluate_torch

    program = AstNode("TsMean", (AstNode("Close"),), {"window": 2})
    result = evaluate_torch(program, tiny_panel, device="cpu")
    reference = evaluate(program, tiny_panel)
    expected = [
        np.nan,
        np.nan,
        10.5,
        20.5,
        np.nan,
        21.5,
        np.nan,
        22.5,
        13.5,
        23.5,
    ]
    np.testing.assert_allclose(result.values, expected, equal_nan=True)
    pd.testing.assert_series_equal(result.values, reference.values)
    pd.testing.assert_series_equal(result.support, reference.support)


def test_torch_cross_section_rank_and_canonical_round_trip(
    tiny_panel: PitPanel,
) -> None:
    import torch

    from mirage_kan.executor import evaluate_torch

    program = AstNode(
        "CSRank",
        (AstNode("Return", (AstNode("Close"),), {"window": 2}),),
    )
    restored = AstNode.from_dict(program.to_dict())
    assert restored.canonical_json() == program.canonical_json()
    assert restored.identity == program.identity

    reference = evaluate(restored, tiny_panel)
    expected = [
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        1.0,
        1.0,
        0.5,
        np.nan,
        1.0,
    ]
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    for device in devices:
        result = evaluate_torch(restored, tiny_panel, device=device)
        np.testing.assert_allclose(result.values, expected, equal_nan=True)
        pd.testing.assert_series_equal(result.values, reference.values)
        pd.testing.assert_series_equal(result.support, reference.support)


def test_torch_scalar_constant_matches_pandas(tiny_panel: PitPanel) -> None:
    from mirage_kan.executor import evaluate_torch

    program = AstNode(
        "Sub",
        (
            AstNode(
                "SafeDiv",
                (
                    AstNode("Volume"),
                    AstNode("TsMean", (AstNode("Volume"),), {"window": 2}),
                ),
            ),
            AstNode("Constant", params={"value": 1.0}),
        ),
    )
    result = evaluate_torch(program, tiny_panel)
    reference = evaluate(program, tiny_panel)
    pd.testing.assert_series_equal(result.values, reference.values)
    pd.testing.assert_series_equal(result.support, reference.support)
