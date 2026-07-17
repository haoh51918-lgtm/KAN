from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from mirage_kan.data import PitPanel


def _panel(days: int = 35, instruments: int = 5) -> tuple[PitPanel, pd.Series]:
    rows: list[dict[str, object]] = []
    dates = pd.bdate_range("2020-01-02", periods=days)
    for date_index, date in enumerate(dates):
        for instrument_index in range(instruments):
            base = 15.0 + instrument_index * 3.0 + date_index * 0.1
            close = base + math.sin((date_index + instrument_index) / 5.0)
            rows.append(
                {
                    "datetime": date,
                    "instrument": f"S{instrument_index}",
                    "open": close * (0.99 + instrument_index * 0.001),
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1000 + 13 * date_index + 29 * instrument_index,
                    "in_universe": True,
                }
            )
    panel = PitPanel.from_frame(pd.DataFrame(rows))
    close = panel.field("Close")
    labels = (close.groupby(level="instrument").shift(-1) / close - 1).rename("fwd")
    return panel, labels


def test_train_only_gp_scorer_physically_excludes_post_train_rows() -> None:
    from mirage_kan.mining.e3 import build_profile_atom_bank
    from mirage_kan.mining.v2_workflow import TrainOnlyRankIcScorer

    panel, labels = _panel()
    cutoff = str(panel.raw.index.get_level_values("datetime").unique()[24].date())
    scorer = TrainOnlyRankIcScorer(panel, labels, ("2020-01-02", cutoff))
    bank = build_profile_atom_bank("short_price")
    ast = bank[0].ast

    score = scorer(ast, object())

    assert np.isfinite(score)
    assert scorer.panel.raw.index.get_level_values("datetime").max() <= pd.Timestamp(
        cutoff
    )
    assert scorer.labels.index.equals(scorer.panel.raw.index)
    assert not hasattr(scorer, "validation")


def test_period_slice_returns_a_physical_panel_and_exact_named_labels() -> None:
    from mirage_kan.mining.v2_workflow import slice_panel_and_labels

    panel, labels = _panel()
    dates = panel.raw.index.get_level_values("datetime").unique()
    sliced_panel, sliced_labels = slice_panel_and_labels(
        panel, labels, (str(dates[5].date()), str(dates[24].date()))
    )

    observed_dates = sliced_panel.raw.index.get_level_values("datetime")
    assert observed_dates.min() == dates[0]
    assert observed_dates.max() == dates[24]
    assert sliced_labels.name == "fwd"
    assert sliced_labels.index.equals(sliced_panel.raw.index)
    assert len(sliced_panel.raw) == 25 * 5
    assert sliced_labels.loc[observed_dates < dates[5]].isna().all()
    assert sliced_labels.loc[observed_dates >= dates[5]].notna().all()

    with pytest.raises(ValueError, match="fwd"):
        slice_panel_and_labels(
            panel, labels.rename("fwd20"), ("2020-01-02", "2020-02-01")
        )
    with pytest.raises(ValueError, match="no panel rows"):
        slice_panel_and_labels(panel, labels, ("2021-01-01", "2021-02-01"))


def test_period_slice_preserves_exact_full_panel_replay_with_sixty_day_warmup() -> None:
    from mirage_kan.dsl import AstNode, evaluate
    from mirage_kan.mining.v2_workflow import slice_panel_and_labels

    panel, labels = _panel(days=140)
    dates = panel.raw.index.get_level_values("datetime").unique()
    start = dates[75]
    end = dates[119]
    sliced_panel, sliced_labels = slice_panel_and_labels(
        panel, labels, (str(start.date()), str(end.date()))
    )
    ast = AstNode("Return", (AstNode("Close"),), {"window": 60})

    full = evaluate(ast, panel)
    replay = evaluate(ast, sliced_panel)
    target_index = sliced_panel.raw.index[
        sliced_panel.raw.index.get_level_values("datetime") >= start
    ]

    assert sliced_panel.raw.index.get_level_values("datetime").min() == dates[15]
    assert sliced_panel.raw.index.get_level_values("datetime").max() == end
    assert (
        sliced_labels.loc[sliced_labels.index.get_level_values("datetime") < start]
        .isna()
        .all()
    )
    pd.testing.assert_series_equal(
        replay.values.loc[target_index], full.values.loc[target_index]
    )
    pd.testing.assert_series_equal(
        replay.support.loc[target_index], full.support.loc[target_index]
    )


def test_profile_run_conversion_recomputes_soft_values_from_final_logits() -> None:
    from mirage_kan.mining.e3 import build_profile_atom_bank
    from mirage_kan.mining.e3_runner import (
        _run_tiny_batch_for_test,
        batched_forward,
        materialize_atom_panel,
    )
    from mirage_kan.mining.v2_workflow import profile_run_candidates

    panel, labels = _panel()
    atom_panel = materialize_atom_panel(panel, "short_price")
    target = (
        labels.reindex(atom_panel.index)
        .to_numpy()
        .reshape(len(atom_panel.dates), len(atom_panel.instruments))
    )
    valid = atom_panel.joint_support & atom_panel.membership & np.isfinite(target)
    run = _run_tiny_batch_for_test(
        profile="short_price",
        atom_values=torch.from_numpy(np.nan_to_num(atom_panel.values)),
        target=torch.from_numpy(np.nan_to_num(target)),
        valid_mask=torch.from_numpy(valid),
        miner_count=2,
        steps=1,
        device="cpu",
    )
    candidates = profile_run_candidates(run, atom_panel)

    assert len(candidates) == 2
    assert candidates[0].method == "kan_e3"
    assert candidates[0].attempt_index == 0
    assert candidates[0].lineage_complete is False
    assert candidates[0].soft_values is not None
    assert candidates[0].soft_values.index.equals(atom_panel.index)
    assert (
        candidates[0]
        .soft_values[
            ~pd.Series(
                atom_panel.joint_support.reshape(-1)
                & atom_panel.membership.reshape(-1),
                index=atom_panel.index,
            )
        ]
        .isna()
        .all()
    )

    logits = torch.stack([miner.final_logits for miner in run.miners])
    expected = (
        batched_forward(
            logits,
            build_profile_atom_bank("short_price"),
            torch.from_numpy(np.nan_to_num(atom_panel.values)),
            tau=0.1,
            mode="soft",
        )[0]
        .numpy()
        .reshape(-1)
    )
    observed = candidates[0].soft_values.to_numpy()
    supported = np.isfinite(observed)
    np.testing.assert_allclose(observed[supported], expected[supported])


def test_gp_attempt_conversion_preserves_all_budget_dispositions() -> None:
    from mirage_kan.mining.gp_control import generate_gp_attempts
    from mirage_kan.mining.v2_workflow import gp_attempt_candidates

    result = generate_gp_attempts(lambda ast, context: 0.01)
    candidates = gp_attempt_candidates(result)

    assert len(candidates) == 256
    assert [candidate.attempt_index for candidate in candidates] == list(range(256))
    assert all(candidate.method == "typed_gp_sr" for candidate in candidates)
    assert sum(candidate.ast is None for candidate in candidates) == sum(
        attempt.ast is None for attempt in result.attempts
    )


def test_label_permutation_is_within_date_membership_only_and_exactly_replays() -> None:
    from mirage_kan.mining.v2_workflow import permute_labels_within_membership

    panel, labels = _panel(days=3, instruments=5)
    membership = panel.membership.copy()
    first_date = membership.index.get_level_values("datetime").min()
    membership.loc[(first_date, "S4")] = False
    panel = PitPanel(
        raw=panel.raw,
        membership=membership,
        observed=panel.observed,
        tradability=panel.tradability,
    )
    original_nonmember = labels.loc[(first_date, "S4")]

    first = permute_labels_within_membership(labels, panel, seed=86028121)
    second = permute_labels_within_membership(labels, panel, seed=86028121)

    pd.testing.assert_series_equal(first, second)
    assert first.name == "fwd"
    assert first.loc[(first_date, "S4")] == original_nonmember
    for date in labels.index.get_level_values("datetime").unique():
        eligible = panel.membership.loc[date] & labels.loc[date].notna()
        assert sorted(first.loc[date][eligible].tolist()) == sorted(
            labels.loc[date][eligible].tolist()
        )


def test_profile_conversion_rejects_wrong_atom_manifest_or_incomplete_population() -> (
    None
):
    from dataclasses import replace

    from mirage_kan.mining.e3_runner import (
        _run_tiny_batch_for_test,
        materialize_atom_panel,
    )
    from mirage_kan.mining.v2_workflow import profile_run_candidates

    panel, _ = _panel()
    atom_panel = materialize_atom_panel(panel, "short_price")
    atoms = torch.from_numpy(np.nan_to_num(atom_panel.values))
    target = atoms[..., 0]
    valid = torch.from_numpy(atom_panel.joint_support & atom_panel.membership)
    run = _run_tiny_batch_for_test(
        profile="short_price",
        atom_values=atoms,
        target=target,
        valid_mask=valid,
        miner_count=1,
        steps=1,
        device="cpu",
    )

    with pytest.raises(ValueError, match="profile"):
        profile_run_candidates(run, replace(atom_panel, profile="long_price"))
    with pytest.raises(ValueError, match="miner indices"):
        profile_run_candidates(
            replace(run, miners=(replace(run.miners[0], miner_index=2),)), atom_panel
        )
