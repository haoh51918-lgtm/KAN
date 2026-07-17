"""Verified point-in-time OHLCV loading without implicit imputation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

RAW_FIELDS = ("open", "high", "low", "close", "volume")
INDEX_NAMES = ("datetime", "instrument")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 of one file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PitPanel:
    """Raw PIT panel with distinct membership, observation, and tradability masks."""

    raw: pd.DataFrame
    membership: pd.Series
    observed: Mapping[str, pd.Series]
    tradability: pd.Series | None
    source_path: Path | None = None
    source_sha256: str | None = None

    @classmethod
    def load(cls, path: Path | str, expected_sha256: str) -> "PitPanel":
        """Load a parquet cache only after verifying its pinned identity."""
        source = Path(path).resolve(strict=True)
        actual = sha256_file(source)
        if actual != expected_sha256:
            raise ValueError(f"PIT cache SHA-256 mismatch: expected {expected_sha256}, got {actual}")
        return cls.from_frame(
            pd.read_parquet(source), source_path=source, source_sha256=actual
        )

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        source_path: Path | None = None,
        source_sha256: str | None = None,
    ) -> "PitPanel":
        """Validate and normalize a frame while preserving every raw missing value."""
        required = {*INDEX_NAMES, *RAW_FIELDS, "in_universe"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"PIT frame missing columns: {missing}")
        if frame.duplicated(list(INDEX_NAMES)).any():
            raise ValueError("PIT frame contains duplicate datetime/instrument rows")

        normalized = frame.loc[:, [*INDEX_NAMES, *RAW_FIELDS]].copy()
        normalized["datetime"] = pd.to_datetime(normalized["datetime"])
        normalized = normalized.set_index(list(INDEX_NAMES)).sort_index()
        normalized.index = normalized.index.set_names(INDEX_NAMES)

        keyed = frame.loc[:, [*INDEX_NAMES, "in_universe"]].copy()
        keyed["datetime"] = pd.to_datetime(keyed["datetime"])
        membership = (
            keyed.set_index(list(INDEX_NAMES))["in_universe"]
            .astype(bool)
            .reindex(normalized.index)
        )
        observed = {
            field: pd.Series(
                np.isfinite(normalized[field].to_numpy(dtype=float)),
                index=normalized.index,
                name=field,
            )
            for field in RAW_FIELDS
        }

        tradability = None
        if "tradable" in frame.columns:
            trade = frame.loc[:, [*INDEX_NAMES, "tradable"]].copy()
            trade["datetime"] = pd.to_datetime(trade["datetime"])
            tradability = (
                trade.set_index(list(INDEX_NAMES))["tradable"]
                .astype("boolean")
                .reindex(normalized.index)
            )
        return cls(
            raw=normalized,
            membership=membership,
            observed=observed,
            tradability=tradability,
            source_path=source_path,
            source_sha256=source_sha256,
        )

    def field(self, name: str) -> pd.Series:
        """Return an unfilled raw field by its DSL leaf name or storage name."""
        field = name.lower()
        if field not in RAW_FIELDS:
            raise KeyError(f"unknown raw field: {name}")
        return self.raw[field].copy()

    def audit(self) -> dict[str, object]:
        """Return machine-readable identity and mask counts."""
        all_raw_missing = ~pd.concat(self.observed, axis=1).any(axis=1)
        dates = self.raw.index.get_level_values("datetime")
        instruments = self.raw.index.get_level_values("instrument")
        return {
            "source_path": str(self.source_path) if self.source_path else None,
            "source_sha256": self.source_sha256,
            "rows": len(self.raw),
            "dates": int(dates.nunique()),
            "date_min": str(dates.min().date()),
            "date_max": str(dates.max().date()),
            "instruments": int(instruments.nunique()),
            "universe_rows": int(self.membership.sum()),
            "in_universe_all_raw_unobserved_rows": int(
                (self.membership & all_raw_missing).sum()
            ),
            "tradability_present": self.tradability is not None,
            "observed_rows": {
                field: int(mask.sum()) for field, mask in self.observed.items()
            },
        }

