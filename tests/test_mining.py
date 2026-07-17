from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from mirage_kan.data import PitPanel
from mirage_kan.dsl import AstNode
from mirage_kan.mining import (
    CandidateScore,
    ast_depth,
    ast_node_count,
    generate_attempts,
    greedy_select,
    permute_labels_within_date,
    score_attempts,
    select_random_control,
)
from mirage_kan.mining.core import MAX_REQUIRED_LAG


PROFILES = ("trend", "mean_reversion", "price_volume", "typed_composition")


def _panel(end: str = "2021-01-08") -> tuple[PitPanel, pd.Series]:
    dates = pd.date_range("2020-12-28", end, freq="D")
    rows = []
    labels = []
    for date_offset, date in enumerate(dates):
        for instrument_offset, instrument in enumerate(("A", "B", "C", "D")):
            close = 10.0 + date_offset + instrument_offset * (1 + date_offset / 10)
            rows.append(
                {
                    "datetime": date,
                    "instrument": instrument,
                    "open": close - 0.2,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "volume": 100 + date_offset * (instrument_offset + 1),
                    "in_universe": True,
                }
            )
            labels.append(
                (date, instrument, float(instrument_offset + date_offset % 2))
            )
    panel = PitPanel.from_frame(pd.DataFrame(rows))
    index = pd.MultiIndex.from_tuples(
        [(date, instrument) for date, instrument, _ in labels],
        names=["datetime", "instrument"],
    )
    return panel, pd.Series([value for _, _, value in labels], index=index, name="fwd")


def _score(
    candidate_id: str,
    profile: str,
    value_offset: float,
    *,
    minimum_score: float,
) -> CandidateScore:
    index = pd.MultiIndex.from_product(
        [pd.date_range("2021-01-01", periods=3), ["A", "B", "C", "D"]],
        names=["datetime", "instrument"],
    )
    base = np.tile(np.arange(4, dtype=float), 3)
    values = pd.Series(base + value_offset, index=index)
    program = AstNode("Return", (AstNode("Close"),), {"window": 2})
    return CandidateScore(
        candidate_id=candidate_id,
        profile=profile,
        attempt_index=0,
        program=program,
        canonical_hash=candidate_id.ljust(64, "0")[:64],
        ast_depth=2,
        ast_nodes=2,
        output_type="dimensionless_ts",
        causal=True,
        unique=True,
        support_rows=len(index),
        eligible_rows=len(index),
        coverage=1.0,
        train_rank_ic=minimum_score,
        validation_rank_ic=minimum_score,
        sign_agreement=True,
        minimum_score=minimum_score,
        eligible=True,
        disposition="eligible",
        values=values,
    )


def test_generation_is_exact_deterministic_and_within_frozen_ast_limits() -> None:
    first = generate_attempts(seed=104729, attempts_per_profile=64)
    second = generate_attempts(seed=104729, attempts_per_profile=64)

    assert len(first) == 256
    assert [attempt.to_record() for attempt in first] == [
        attempt.to_record() for attempt in second
    ]
    assert {
        profile: sum(a.profile == profile for a in first) for profile in PROFILES
    } == {profile: 64 for profile in PROFILES}
    for attempt in first:
        assert attempt.program is not None
        assert attempt.program.validate().lookback <= MAX_REQUIRED_LAG + 1
        assert ast_depth(attempt.program) <= 6
        assert ast_node_count(attempt.program) <= 20


def test_scoring_rejects_any_panel_past_validation_boundary() -> None:
    panel, labels = _panel("2022-01-02")
    attempts = generate_attempts(seed=104729, attempts_per_profile=1)

    with pytest.raises(ValueError, match="validation boundary"):
        score_attempts(
            attempts,
            panel,
            labels,
            train=("2020-01-01", "2020-12-31"),
            validation=("2021-01-01", "2021-12-31"),
            minimum_coverage=0.0,
            minimum_absolute_rank_ic=0.0,
            label_mode="observed",
        )


def test_within_date_permutation_repeats_scoring_with_distinct_provenance() -> None:
    panel, labels = _panel()
    attempts = generate_attempts(seed=104729, attempts_per_profile=2)
    permuted = permute_labels_within_date(labels, seed=314159)

    for date in labels.index.get_level_values("datetime").unique():
        original = labels.xs(date, level="datetime").sort_values().to_numpy()
        shuffled = permuted.xs(date, level="datetime").sort_values().to_numpy()
        np.testing.assert_array_equal(original, shuffled)
    assert not labels.equals(permuted)

    observed = score_attempts(
        attempts,
        panel,
        labels,
        train=("2020-01-01", "2020-12-31"),
        validation=("2021-01-01", "2021-12-31"),
        minimum_coverage=0.0,
        minimum_absolute_rank_ic=0.0,
        label_mode="observed",
    )
    null = score_attempts(
        attempts,
        panel,
        permuted,
        train=("2020-01-01", "2020-12-31"),
        validation=("2021-01-01", "2021-12-31"),
        minimum_coverage=0.0,
        minimum_absolute_rank_ic=0.0,
        label_mode="within_date_permutation",
    )
    assert observed is not null
    assert observed.scoring_run_id != null.scoring_run_id
    assert observed.label_sha256 != null.label_sha256
    assert null.label_mode == "within_date_permutation"
    for candidate in (*observed.candidates, *null.candidates):
        if candidate.program is not None:
            dates = candidate.values.index.get_level_values("datetime")
            assert dates.min() >= pd.Timestamp("2021-01-01")
            assert len(candidate.values) == 8 * 4


def test_greedy_selection_enforces_diversity_and_profile_quota_without_padding() -> (
    None
):
    trend = _score("a", "trend", 0.0, minimum_score=0.09)
    duplicate_signal = replace(
        _score("b", "mean_reversion", 0.0, minimum_score=0.08),
        values=trend.values.copy(),
    )
    mean_reversion = _score("c", "mean_reversion", 0.0, minimum_score=0.07)
    mean_reversion = replace(
        mean_reversion,
        values=pd.Series(
            np.tile([0.0, 2.0, 3.0, 1.0], 3), index=mean_reversion.values.index
        ),
    )
    price_volume = _score("d", "price_volume", 0.0, minimum_score=0.06)
    price_volume = replace(
        price_volume,
        values=pd.Series(
            np.tile([0.0, 3.0, 1.0, 2.0], 3), index=price_volume.values.index
        ),
    )

    result = greedy_select(
        [trend, duplicate_signal, mean_reversion, price_volume],
        library_cap=16,
        minimum_library_size=8,
        minimum_profiles=3,
        maximum_absolute_spearman=0.80,
    )

    assert [candidate.candidate_id for candidate in result.selected] == ["a", "c", "d"]
    assert result.by_candidate["b"] == "rejected_diversity"
    assert result.minimum_size_met is False
    assert result.profile_quota_met is True
    assert len(result.selected) == 3


def test_undefined_pairwise_spearman_is_not_admitted_as_diverse() -> None:
    constant = replace(
        _score("a", "trend", 0.0, minimum_score=0.02),
        values=pd.Series(
            np.ones(12), index=_score("a", "trend", 0.0, minimum_score=0.02).values.index
        ),
    )
    varying = _score("b", "mean_reversion", 0.0, minimum_score=0.01)

    result = greedy_select(
        [constant, varying],
        library_cap=2,
        minimum_library_size=1,
        minimum_profiles=1,
        maximum_absolute_spearman=0.80,
    )

    assert [candidate.candidate_id for candidate in result.selected] == ["a"]
    assert result.by_candidate["b"] == "rejected_diversity_undefined"


def test_random_control_is_deterministic_and_has_no_label_metrics() -> None:
    panel, _ = _panel()
    attempts = generate_attempts(seed=104729, attempts_per_profile=4)
    first = select_random_control(
        attempts, panel, seed=8675309, library_cap=8, minimum_coverage=0.0
    )
    second = select_random_control(
        attempts, panel, seed=8675309, library_cap=8, minimum_coverage=0.0
    )

    assert [candidate.canonical_hash for candidate in first] == [
        candidate.canonical_hash for candidate in second
    ]
    assert len(first) <= 8
    assert all(candidate.train_rank_ic is None for candidate in first)
    assert all(candidate.validation_rank_ic is None for candidate in first)
    assert all(
        candidate.disposition == "random_label_free_eligible" for candidate in first
    )
