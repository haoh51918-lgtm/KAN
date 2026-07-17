from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch


def _tiny_inputs(profile: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from mirage_kan.mining.e3 import build_profile_atom_bank

    atom_count = len(build_profile_atom_bank(profile))
    generator = torch.Generator().manual_seed(20260717)
    atoms = torch.randn(25, 5, atom_count, generator=generator, dtype=torch.float64)
    target = atoms[..., 0] - 0.3 * atoms[..., 3]
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[3, 4] = False
    return atoms, target, mask


def test_two_unit_mlp_is_deterministic_and_capacity_matched_for_every_profile() -> None:
    from mirage_kan.mining.e3 import PROFILE_SPECS, build_profile_atom_bank
    from mirage_kan.mining.mlp_control import MatchedBlackboxMLP

    for profile_index, profile in enumerate(PROFILE_SPECS):
        global_attempt_index = profile_index * 64 + 7
        first = MatchedBlackboxMLP(profile, global_attempt_index)
        second = MatchedBlackboxMLP(profile, global_attempt_index)
        atom_count = len(build_profile_atom_bank(profile))

        assert first.hidden_width == 2
        assert first.seed == 32452843 + global_attempt_index
        assert first.kan_parameter_count == 2 * atom_count
        assert first.parameter_count == 2 * atom_count + 5
        assert first.parameter_relative_gap <= 0.10
        assert isinstance(first.network[0], torch.nn.Linear)
        assert isinstance(first.network[1], torch.nn.SiLU)
        assert isinstance(first.network[2], torch.nn.Linear)
        for left, right in zip(first.parameters(), second.parameters(), strict=True):
            torch.testing.assert_close(left, right)


def test_tiny_control_replays_exact_pairing_and_complete_training_receipt() -> None:
    from mirage_kan.mining.e3 import atom_manifest_sha256, build_profile_atom_bank
    from mirage_kan.mining.e3_runner import draw_training_bootstrap
    from mirage_kan.mining.mlp_control import _run_tiny_matched_control_for_test

    atoms, target, mask = _tiny_inputs("short_price")
    bootstrap = draw_training_bootstrap(25, 49979687 + 11)
    first = _run_tiny_matched_control_for_test(
        profile="short_price",
        kan_global_attempt_index=11,
        bootstrap=bootstrap,
        atom_values=atoms,
        target=target,
        valid_mask=mask,
        steps=2,
        device="cpu",
    )
    second = _run_tiny_matched_control_for_test(
        profile="short_price",
        kan_global_attempt_index=11,
        bootstrap=bootstrap,
        atom_values=atoms,
        target=target,
        valid_mask=mask,
        steps=2,
        device="cpu",
    )

    assert first.profile == "short_price"
    assert first.kan_global_attempt_index == 11
    assert first.seed == 32452854
    assert first.bootstrap == bootstrap
    assert first.optimizer == "Adam"
    assert first.learning_rate == 0.03
    assert first.scheduled_updates == 300
    assert first.completed_updates == 2
    assert len(first.trajectory) == 2
    assert first.initial_parameters.shape == (69,)
    assert first.final_parameters.shape == (69,)
    assert first.first_step_data_gradient.shape == (69,)
    assert first.prediction.shape == target.shape
    torch.testing.assert_close(first.prediction, second.prediction)
    torch.testing.assert_close(first.initial_parameters, second.initial_parameters)
    torch.testing.assert_close(first.final_parameters, second.final_parameters)
    torch.testing.assert_close(
        first.first_step_data_gradient, second.first_step_data_gradient
    )
    for left, right in zip(first.trajectory, second.trajectory, strict=True):
        assert left.update_index == right.update_index
        assert left.total_loss == pytest.approx(right.total_loss)
        assert left.mean_daily_ic == pytest.approx(right.mean_daily_ic)
        torch.testing.assert_close(left.parameters, right.parameters)
    assert not torch.equal(first.initial_parameters, first.final_parameters)
    assert torch.isfinite(first.first_step_data_gradient).all()
    assert first.valid_support_sha256 == second.valid_support_sha256
    assert first.atom_manifest_sha256 == second.atom_manifest_sha256
    assert first.atom_manifest_sha256 == atom_manifest_sha256(
        build_profile_atom_bank("short_price")
    )


def test_control_output_is_explicitly_nonpromotable_and_not_a_factor_library() -> None:
    from mirage_kan.mining.e3_runner import draw_training_bootstrap
    from mirage_kan.mining.mlp_control import (
        MatchedBlackboxControlPanel,
        _run_tiny_matched_control_for_test,
    )

    atoms, target, mask = _tiny_inputs("short_price")
    control = _run_tiny_matched_control_for_test(
        profile="short_price",
        kan_global_attempt_index=3,
        bootstrap=draw_training_bootstrap(25, 49979690),
        atom_values=atoms,
        target=target,
        valid_mask=mask,
        steps=1,
        device="cpu",
    )
    panel = MatchedBlackboxControlPanel((control,))

    assert panel.role == "falsification_control_never_production"
    assert panel.output_kind == "control_panel_not_factor_library"
    assert panel.promotion_eligible is False
    assert panel.factor_library_publication_allowed is False
    assert control.kan_mined is False
    assert control.promotion_eligible is False
    assert control.factor_library_publication_allowed is False


def test_completed_control_replays_final_parameters_on_a_full_atom_panel() -> None:
    from mirage_kan.mining.e3 import build_profile_atom_bank
    from mirage_kan.mining.e3_runner import AtomPanel, draw_training_bootstrap
    from mirage_kan.mining.mlp_control import (
        MatchedBlackboxMLP,
        _run_tiny_matched_control_for_test,
        replay_control_on_atom_panel,
    )

    atoms, target, mask = _tiny_inputs("short_price")
    tiny = _run_tiny_matched_control_for_test(
        profile="short_price",
        kan_global_attempt_index=5,
        bootstrap=draw_training_bootstrap(25, 49979692),
        atom_values=atoms,
        target=target,
        valid_mask=mask,
        steps=1,
        device="cpu",
    )
    step = tiny.trajectory[0]
    complete = replace(
        tiny,
        completed_updates=300,
        trajectory=tuple(replace(step, update_index=index) for index in range(300)),
    )
    dates = pd.bdate_range("2022-01-03", periods=25)
    instruments = pd.Index([f"S{index}" for index in range(5)], name="instrument")
    index = pd.MultiIndex.from_product(
        (dates, instruments), names=("datetime", "instrument")
    )
    manifest = build_profile_atom_bank("short_price")
    atom_panel = AtomPanel(
        profile="short_price",
        atom_manifest=manifest,
        dates=dates,
        instruments=instruments,
        index=index,
        values=atoms.numpy(),
        support=np.ones(atoms.shape, dtype=bool),
        joint_support=np.ones(target.shape, dtype=bool),
        membership=np.ones(target.shape, dtype=bool),
    )

    replay = replay_control_on_atom_panel(complete, atom_panel)

    model = MatchedBlackboxMLP("short_price", 5)
    offset = 0
    with torch.no_grad():
        for parameter in model.parameters():
            count = parameter.numel()
            parameter.copy_(
                complete.final_parameters[offset : offset + count].reshape(
                    parameter.shape
                )
            )
            offset += count
        expected = model(atoms).numpy().reshape(-1)
    np.testing.assert_allclose(replay.to_numpy(), expected)
    assert replay.index.equals(index)

    with pytest.raises(ValueError, match="complete 300"):
        replay_control_on_atom_panel(tiny, atom_panel)


def test_multiplicity_weights_are_date_weights_not_row_replication_shortcuts() -> None:
    from mirage_kan.mining.mlp_control import _weighted_mean_daily_ic

    score = torch.tensor(
        [[0.0, 1.0, 2.0], [2.0, 1.0, 0.0], [0.0, 1.0, 2.0]],
        dtype=torch.float64,
    )
    target = torch.tensor(
        [[0.0, 2.0, 4.0], [0.0, 1.0, 2.0], [2.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    mask = torch.ones_like(target, dtype=torch.bool)
    weights = torch.tensor([2.0, 1.0, 0.0], dtype=torch.float64)

    # Daily ICs are +1, -1, -1; the zero-weight date must not enter the mean.
    assert float(
        _weighted_mean_daily_ic(score, target, mask, weights)
    ) == pytest.approx(1.0 / 3.0)


def test_scientific_entrypoint_exposes_no_budget_or_architecture_knobs() -> None:
    from mirage_kan.mining.mlp_control import run_matched_blackbox_controls

    assert tuple(inspect.signature(run_matched_blackbox_controls).parameters) == (
        "panel",
        "target",
        "pairings",
        "minimum_library_size",
        "library_cap",
        "device",
    )
    source = inspect.getsource(run_matched_blackbox_controls)
    assert "_run_tiny_matched_control_for_test" not in source
    assert (
        "hidden_width"
        not in inspect.signature(run_matched_blackbox_controls).parameters
    )
    assert "steps" not in inspect.signature(run_matched_blackbox_controls).parameters

    source = inspect.getsource(run_matched_blackbox_controls)
    assert "2020-12-31" in source


def test_control_uses_the_same_objective_dates_as_paired_kan_after_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mirage_kan.mining.mlp_control as control_module
    from mirage_kan.mining.e3_runner import draw_training_bootstrap

    from tests.mining.test_e3_runner import _synthetic_panel

    panel, target = _synthetic_panel(days=90)
    dates = pd.DatetimeIndex(
        target.index.get_level_values("datetime").unique()
    ).sort_values()
    target = target.copy()
    target.loc[target.index.get_level_values("datetime").isin(dates[:60])] = np.nan
    target.loc[target.index.get_level_values("datetime").isin(dates[-5:])] = np.nan
    pairings = tuple(
        control_module.MLPControlPairing(
            "short_price",
            attempt,
            draw_training_bootstrap(25, 49979687 + attempt),
        )
        for attempt in range(8)
    )
    observed: list[tuple[int, int]] = []

    def capture_training_inputs(**kwargs: object) -> int:
        atom_values = kwargs["atom_values"]
        bootstrap = kwargs["bootstrap"]
        assert isinstance(atom_values, torch.Tensor)
        observed.append((atom_values.shape[0], bootstrap.date_count))
        return int(kwargs["kan_global_attempt_index"])

    monkeypatch.setattr(control_module, "_train_control", capture_training_inputs)

    result = control_module.run_matched_blackbox_controls(
        panel,
        target,
        pairings,
        minimum_library_size=6,
        library_cap=16,
    )

    assert result.controls == tuple(range(8))
    assert observed == [(25, 25)] * 8


def test_illegal_pairing_shapes_bootstrap_and_values_fail_closed() -> None:
    from mirage_kan.mining.e3_runner import draw_training_bootstrap
    from mirage_kan.mining.mlp_control import (
        MLPControlPairing,
        MatchedBlackboxMLP,
        _run_tiny_matched_control_for_test,
    )

    with pytest.raises(ValueError, match="profile segment"):
        MatchedBlackboxMLP("short_price", 64)
    with pytest.raises(ValueError, match="global attempt"):
        MatchedBlackboxMLP("short_price", -1)

    atoms, target, mask = _tiny_inputs("short_price")
    good = draw_training_bootstrap(25, 49979687)
    with pytest.raises(ValueError, match="exact paired KAN bootstrap"):
        _run_tiny_matched_control_for_test(
            profile="short_price",
            kan_global_attempt_index=0,
            bootstrap=draw_training_bootstrap(25, 49979688),
            atom_values=atoms,
            target=target,
            valid_mask=mask,
            steps=1,
            device="cpu",
        )
    with pytest.raises(ValueError, match="shape"):
        _run_tiny_matched_control_for_test(
            profile="short_price",
            kan_global_attempt_index=0,
            bootstrap=good,
            atom_values=atoms[..., :-1],
            target=target,
            valid_mask=mask,
            steps=1,
            device="cpu",
        )
    bad = atoms.clone()
    bad[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        _run_tiny_matched_control_for_test(
            profile="short_price",
            kan_global_attempt_index=0,
            bootstrap=good,
            atom_values=bad,
            target=target,
            valid_mask=mask,
            steps=1,
            device="cpu",
        )
    with pytest.raises(ValueError, match="scheduled budget"):
        _run_tiny_matched_control_for_test(
            profile="short_price",
            kan_global_attempt_index=0,
            bootstrap=good,
            atom_values=atoms,
            target=target,
            valid_mask=mask,
            steps=301,
            device="cpu",
        )
    with pytest.raises(ValueError, match="unique"):
        MLPControlPairing.validate_many(
            (
                MLPControlPairing("short_price", 0, good),
                MLPControlPairing("short_price", 0, good),
            ),
            minimum_library_size=6,
            library_cap=16,
        )


def test_v5_mlp_pairing_bounds_accept_six_and_reject_five_or_seventeen() -> None:
    from mirage_kan.mining.e3_runner import draw_training_bootstrap
    from mirage_kan.mining.mlp_control import MLPControlPairing

    pairings = tuple(
        MLPControlPairing(
            "short_price",
            attempt,
            draw_training_bootstrap(25, 49979687 + attempt),
        )
        for attempt in range(17)
    )

    accepted = MLPControlPairing.validate_many(
        pairings[:6], minimum_library_size=6, library_cap=16
    )
    assert accepted == pairings[:6]
    for invalid in (pairings[:5], pairings):
        with pytest.raises(ValueError, match="frozen admission bounds"):
            MLPControlPairing.validate_many(
                invalid, minimum_library_size=6, library_cap=16
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_tiny_smoke() -> None:
    from mirage_kan.mining.e3_runner import draw_training_bootstrap
    from mirage_kan.mining.mlp_control import _run_tiny_matched_control_for_test

    atoms, target, mask = _tiny_inputs("price_volume")
    result = _run_tiny_matched_control_for_test(
        profile="price_volume",
        kan_global_attempt_index=192,
        bootstrap=draw_training_bootstrap(25, 49979687 + 192),
        atom_values=atoms,
        target=target,
        valid_mask=mask,
        steps=1,
        device="cuda",
    )
    assert result.prediction.device.type == "cpu"
    assert result.final_parameters.device.type == "cpu"
