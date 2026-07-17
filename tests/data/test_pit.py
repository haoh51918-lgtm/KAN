from __future__ import annotations

import numpy as np
import pandas as pd

from mirage_kan.data.pit import PitPanel


def test_membership_and_observation_are_distinct() -> None:
    frame = pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2020-01-01")],
            "instrument": ["A"],
            "open": [np.nan],
            "high": [np.nan],
            "low": [np.nan],
            "close": [np.nan],
            "volume": [np.nan],
            "in_universe": [True],
        }
    )
    panel = PitPanel.from_frame(frame)
    assert bool(panel.membership.iloc[0])
    assert not bool(panel.observed["close"].iloc[0])
    assert panel.tradability is None


def test_raw_missing_values_are_not_filled(tiny_panel: PitPanel) -> None:
    value = tiny_panel.field("close")
    assert np.isnan(value.loc[(pd.Timestamp("2020-01-03"), "A")])

