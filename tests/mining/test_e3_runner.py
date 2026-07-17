from __future__ import annotations

import inspect
import math

import numpy as np
import pandas as pd
import pytest
import torch

from mirage_kan.data import PitPanel
from mirage_kan.dsl import evaluate


def _synthetic_panel(
    days: int = 30, instruments: int = 4
) -> tuple[PitPanel, pd.Series]:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2020-01-01", periods=days, freq="B")
    for date_index, date in enumerate(dates):
        for instrument_index in range(instruments):
            base = 20.0 + 2.0 * instrument_index + 0.15 * date_index
            close = base + math.sin((date_index + instrument_index) / 4.0)
            rows.append(
                {
                    "datetime": date,
                    "instrument": f"S{instrument_index}",
                    "open": close * (0.995 + 0.001 * instrument_index),
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1000.0 + 11.0 * date_index + 37.0 * instrument_index,
                    "in_universe": not (date_index == 14 and instrument_index == 3),
                }
            )
    frame = pd.DataFrame(rows)
    frame.loc[
        (frame["datetime"] == dates[12]) & (frame["instrument"] == "S2"),
        "open",
    ] = np.nan
    panel = PitPanel.from_frame(frame)
    close = panel.field("Close")
    target = (
        close.groupby(level="instrument", sort=False).shift(-1) / close - 1.0
    ).rename("fwd")
    return panel, target


def test_atom_panel_is_direct_public_dsl_evaluation_with_joint_support() -> None:
    from mirage_kan.mining.e3_runner import materialize_atom_panel

    panel, _ = _synthetic_panel()
    result = materialize_atom_panel(panel, "short_price")

    assert result.values.shape == (30, 4, 32)
    assert result.support.shape == result.values.shape
    assert result.joint_support.shape == (30, 4)
    assert result.index.equals(
        pd.MultiIndex.from_product(
            (result.dates, result.instruments), names=("datetime", "instrument")
        )
    )
    for atom in result.atom_manifest:
        reference = evaluate(atom.ast, panel)
        flat_values = result.values[..., atom.atom_index].reshape(-1)
        flat_support = result.support[..., atom.atom_index].reshape(-1)
        np.testing.assert_allclose(
            flat_values,
            reference.values.reindex(result.index).to_numpy(dtype=float),
            equal_nan=True,
        )
        np.testing.assert_array_equal(
            flat_support,
            reference.support.reindex(result.index, fill_value=False).to_numpy(bool),
        )
    np.testing.assert_array_equal(result.joint_support, result.support.all(axis=-1))
    assert np.isnan(result.values[~result.support]).all()


def test_pcg64_nonwrapping_block_bootstrap_has_literal_and_exact_replay() -> None:
    from mirage_kan.mining.e3_runner import draw_training_bootstrap

    receipt = draw_training_bootstrap(date_count=25, seed=49979687)
    expected_start = int(np.random.Generator(np.random.PCG64(49979687)).integers(0, 6))
    expected_indices = tuple(range(expected_start, expected_start + 20))

    assert receipt.target_date_draws == 20
    assert receipt.block_length == 20
    assert receipt.drawn_block_starts == (expected_start,)
    assert receipt.sampled_date_indices == expected_indices
    assert receipt.multiplicities == tuple(
        int(value) for value in np.bincount(expected_indices, minlength=25)
    )
    assert receipt == draw_training_bootstrap(date_count=25, seed=49979687)
    assert all(0 <= index < 25 for index in receipt.sampled_date_indices)

    repeated = draw_training_bootstrap(date_count=41, seed=0)
    assert repeated.drawn_block_starts == (18, 14)
    assert repeated.sampled_date_indices == tuple([*range(18, 38), *range(14, 27)])
    assert repeated.target_date_draws == 33
    assert sum(repeated.multiplicities) == 33
    assert max(repeated.multiplicities) == 2


@pytest.mark.parametrize("mode", ["soft", "hard_st"])
def test_batched_forward_and_objective_match_single_miner(mode: str) -> None:
    from mirage_kan.mining.e3 import CategoricalE3KAN, training_objective
    from mirage_kan.mining.e3_runner import (
        batched_forward,
        batched_training_objective,
    )

    model = CategoricalE3KAN("short_price", seed=730001)
    generator = torch.Generator().manual_seed(19)
    atoms = torch.randn(
        5, 4, model.atom_count, generator=generator, dtype=torch.float64
    )
    target = atoms[..., 0] - 0.3 * atoms[..., 3]
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[0, 0] = False
    weights = torch.ones(1, atoms.shape[0], dtype=torch.float64)

    single = training_objective(
        model, atoms, target, tau=0.7, mode=mode, valid_mask=mask
    )
    batched_score = batched_forward(
        model.gate_logits[None], model.atom_manifest, atoms, tau=0.7, mode=mode
    )
    batched = batched_training_objective(
        model.gate_logits[None],
        model.atom_manifest,
        atoms,
        target,
        mask,
        weights,
        tau=0.7,
        mode=mode,
    )

    torch.testing.assert_close(batched_score[0], single["score"])
    for name in ("total_loss", "mean_daily_ic", "entropy", "edge_overlap"):
        torch.testing.assert_close(batched[name][0], single[name])


def test_tiny_training_replays_seeds_bootstraps_and_all_receipts() -> None:
    from mirage_kan.mining.e3_runner import _run_tiny_batch_for_test

    generator = torch.Generator().manual_seed(31)
    atoms = torch.randn(25, 4, 32, generator=generator, dtype=torch.float64)
    target = atoms[..., 0] - 0.2 * atoms[..., 8]
    mask = torch.ones_like(target, dtype=torch.bool)
    first = _run_tiny_batch_for_test(
        profile="short_price",
        atom_values=atoms,
        target=target,
        valid_mask=mask,
        miner_count=64,
        steps=1,
        device="cpu",
    )
    second = _run_tiny_batch_for_test(
        profile="short_price",
        atom_values=atoms,
        target=target,
        valid_mask=mask,
        miner_count=64,
        steps=1,
        device="cpu",
    )

    assert len(first.miners) == 64
    assert [miner.miner_seed for miner in first.miners] == list(range(730001, 730065))
    assert [miner.bootstrap.seed for miner in first.miners] == list(
        range(49979687, 49979751)
    )
    assert all(len(miner.trajectory) == 1 for miner in first.miners)
    assert all(
        miner.first_step_data_gradient.shape == (2, 32) for miner in first.miners
    )
    assert all(miner.candidate_ast.op == "Sub" for miner in first.miners)
    for left, right in zip(first.miners, second.miners, strict=True):
        torch.testing.assert_close(left.initial_logits, right.initial_logits)
        torch.testing.assert_close(left.final_logits, right.final_logits)
        torch.testing.assert_close(
            left.first_step_data_gradient, right.first_step_data_gradient
        )
        assert left.bootstrap == right.bootstrap
        assert left.hardening == right.hardening
        assert left.fidelity == right.fidelity


def test_low_margin_and_same_top_atom_are_recorded_without_hidden_retry() -> None:
    from mirage_kan.mining.e3_runner import _run_tiny_batch_for_test

    atoms = torch.randn(
        25, 3, 32, generator=torch.Generator().manual_seed(71), dtype=torch.float64
    )
    target = atoms[..., 1]
    logits = torch.zeros(1, 2, 32, dtype=torch.float64)
    logits[0, :, 0] = 1e-9
    run = _run_tiny_batch_for_test(
        profile="short_price",
        atom_values=atoms,
        target=target,
        valid_mask=torch.ones_like(target, dtype=torch.bool),
        miner_count=1,
        steps=0,
        device="cpu",
        initial_logits=logits,
    )

    assert len(run.miners) == 1
    miner = run.miners[0]
    assert miner.hardening.positive.atom_index != miner.hardening.negative.atom_index
    assert any(
        item.reason == "same_atom_cancellation"
        for item in miner.hardening.rejected_alternates
    )
    assert min(miner.hardening.positive.margin, miner.hardening.negative.margin) < 0.05
    assert "low_gate_margin" in miner.admission_failures


def test_scientific_entrypoint_has_no_scientific_budget_knobs() -> None:
    from mirage_kan.mining.e3_runner import run_e3_profile

    assert tuple(inspect.signature(run_e3_profile).parameters) == (
        "panel",
        "target",
        "profile",
        "device",
    )
    assert "_run_tiny_batch_for_test" not in inspect.getsource(run_e3_profile)


def test_scientific_entrypoint_bootstraps_only_objective_dates_after_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mirage_kan.mining.e3_runner as runner

    panel, target = _synthetic_panel(days=90)
    dates = pd.DatetimeIndex(
        target.index.get_level_values("datetime").unique()
    ).sort_values()
    target = target.copy()
    target.loc[target.index.get_level_values("datetime").isin(dates[:60])] = np.nan
    target.loc[target.index.get_level_values("datetime").isin(dates[-5:])] = np.nan
    observed: dict[str, object] = {}

    def capture_training_inputs(**kwargs: object) -> str:
        observed.update(kwargs)
        atom_values = kwargs["atom_values"]
        assert isinstance(atom_values, torch.Tensor)
        observed["bootstrap"] = runner._bootstrap_receipts(
            "short_price", 1, atom_values.shape[0]
        )[0]
        return "captured"

    monkeypatch.setattr(runner, "_train_batch", capture_training_inputs)

    assert runner.run_e3_profile(panel, target, "short_price") == "captured"
    assert observed["atom_values"].shape[0] == 25
    assert observed["target"].shape[0] == 25
    assert observed["valid_mask"].shape[0] == 25
    assert observed["prediction_mask"].shape[0] == 25
    assert observed["bootstrap"].date_count == 25

    full_atoms = runner.materialize_atom_panel(panel, "short_price")
    np.testing.assert_allclose(
        observed["atom_values"][0].numpy(),
        np.where(full_atoms.support[60], full_atoms.values[60], 0.0),
    )


def test_scientific_entrypoint_rejects_validation_rows() -> None:
    from mirage_kan.mining.e3_runner import run_e3_profile

    panel, target = _synthetic_panel()
    shifted = panel.raw.reset_index()
    shifted["datetime"] = shifted["datetime"] + pd.DateOffset(years=1)
    shifted["in_universe"] = panel.membership.to_numpy()
    validation_panel = PitPanel.from_frame(shifted)
    validation_target = target.copy()
    validation_target.index = pd.MultiIndex.from_arrays(
        [
            target.index.get_level_values("datetime") + pd.DateOffset(years=1),
            target.index.get_level_values("instrument"),
        ],
        names=target.index.names,
    )

    with pytest.raises(ValueError, match="train boundary"):
        run_e3_profile(validation_panel, validation_target, "short_price")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_tiny_smoke() -> None:
    from mirage_kan.mining.e3_runner import _run_tiny_batch_for_test

    atoms = torch.randn(25, 3, 32, dtype=torch.float64)
    target = atoms[..., 0] - atoms[..., 1]
    result = _run_tiny_batch_for_test(
        profile="short_price",
        atom_values=atoms,
        target=target,
        valid_mask=torch.ones_like(target, dtype=torch.bool),
        miner_count=2,
        steps=1,
        device="cuda",
    )
    assert len(result.miners) == 2
    assert all(miner.final_logits.device.type == "cpu" for miner in result.miners)
