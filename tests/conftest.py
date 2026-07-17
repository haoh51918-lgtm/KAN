from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mirage_kan.data.pit import PitPanel


@pytest.fixture
def tiny_panel() -> PitPanel:
    rows = []
    for instrument, base in (("A", 10.0), ("B", 20.0)):
        for offset, date in enumerate(pd.date_range("2020-01-01", periods=5)):
            close = base + offset
            rows.append(
                {
                    "datetime": date,
                    "instrument": instrument,
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 100.0 + offset,
                    "in_universe": True,
                }
            )
    frame = pd.DataFrame(rows)
    frame.loc[(frame["instrument"] == "A") & (frame["datetime"] == pd.Timestamp("2020-01-03")), "close"] = np.nan
    return PitPanel.from_frame(frame)

