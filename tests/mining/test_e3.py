from __future__ import annotations

import inspect

import pytest
import torch

from mirage_kan.dsl import AstNode, DslType


EXPECTED_PROFILE_SHAPES = {
    "short_price": {
        "windows": {2, 3, 5, 10},
        "fields": {"Open", "High", "Low", "Close"},
        "families": {"return", "price_vs_mean"},
        "count": 32,
    },
    "long_price": {
        "windows": {10, 20, 40, 60},
        "fields": {"Open", "High", "Low", "Close"},
        "families": {"return", "price_vs_mean"},
        "count": 32,
    },
    "reversal": {
        "windows": {2, 3, 5, 10, 20},
        "fields": {"Open", "High", "Low", "Close"},
        "families": {"mean_vs_price", "lag_vs_price"},
        "count": 40,
    },
    "price_volume": {
        "windows": {2, 3, 5, 10, 20, 40, 60},
        "fields": {"Open", "High", "Low", "Close", "Volume"},
        "families": {"return", "volume_change", "volume_vs_mean"},
        "count": 42,
    },
}


def test_profile_atom_banks_are_exact_typed_and_unique() -> None:
    from mirage_kan.mining.e3 import build_profile_atom_bank

    for profile, expected in EXPECTED_PROFILE_SHAPES.items():
        first = build_profile_atom_bank(profile)
        second = build_profile_atom_bank(profile)

        assert first == second
        assert len(first) == expected["count"]
        assert {atom.family for atom in first} == expected["families"]
        assert {atom.window for atom in first} == expected["windows"]
        assert {atom.field for atom in first} == expected["fields"]
        assert [atom.atom_index for atom in first] == list(range(len(first)))
        assert [atom.canonical_hash for atom in first] == sorted(
            atom.canonical_hash for atom in first
        )
        assert len({atom.canonical_hash for atom in first}) == len(first)
        assert len({atom.ast.canonical_json() for atom in first}) == len(first)
        for atom in first:
            contract = atom.ast.validate()
            assert contract.output_type is DslType.DIMENSIONLESS_TS
            assert contract.causal is True
            assert atom.canonical_hash == atom.ast.identity


def test_profile_atom_family_shapes_match_frozen_financial_semantics() -> None:
    from mirage_kan.mining.e3 import build_profile_atom_bank

    bank = build_profile_atom_bank("price_volume")
    assert sum(atom.family == "return" for atom in bank) == 28
    assert sum(atom.family == "volume_change" for atom in bank) == 7
    assert sum(atom.family == "volume_vs_mean" for atom in bank) == 7
    assert all(atom.field != "Volume" for atom in bank if atom.family == "return")
    assert all(atom.field == "Volume" for atom in bank if atom.family != "return")

    lag_atom = next(
        atom
        for atom in build_profile_atom_bank("reversal")
        if atom.family == "lag_vs_price" and atom.field == "Close" and atom.window == 5
    )
    close = AstNode("Close")
    lag = AstNode("Delay", (close,), {"window": 5})
    expected = AstNode("SafeDiv", (AstNode("Sub", (lag, close)), lag))
    assert lag_atom.ast.canonical_json() == expected.canonical_json()


def test_public_model_accepts_profile_and_integer_seed_not_a_gp_ast() -> None:
    from mirage_kan.mining.e3 import CategoricalE3KAN

    parameters = inspect.signature(CategoricalE3KAN).parameters
    assert set(parameters) == {"profile", "seed", "dtype"}
    assert all("ast" not in name and "program" not in name for name in parameters)

    first = CategoricalE3KAN("short_price", seed=730001)
    second = CategoricalE3KAN("short_price", seed=730001)
    torch.testing.assert_close(first.gate_logits, second.gate_logits)


def test_soft_edges_have_nonzero_gradients_on_synthetic_tensor() -> None:
    from mirage_kan.mining.e3 import CategoricalE3KAN, training_objective

    model = CategoricalE3KAN("short_price", seed=730001)
    generator = torch.Generator().manual_seed(17)
    atoms = torch.randn(
        5, 7, model.atom_count, generator=generator, dtype=torch.float64
    )
    target = atoms[..., 0] - 0.4 * atoms[..., 3] + 0.1 * atoms[..., 7]

    objective = training_objective(model, atoms, target, tau=1.25, mode="soft")
    objective["total_loss"].backward()

    assert objective["score"].shape == target.shape
    assert model.gate_logits.grad is not None
    assert torch.isfinite(model.gate_logits.grad).all()
    assert torch.all(model.gate_logits.grad.abs().sum(dim=1) > 0)
    assert objective["mean_daily_ic"].isfinite()


def test_hard_st_forward_equals_independent_hard_ast_replay() -> None:
    from mirage_kan.mining.e3 import (
        CategoricalE3KAN,
        evaluate_hard_ast_from_atoms,
        harden_checkpoint,
    )

    model = CategoricalE3KAN("short_price", seed=730001)
    with torch.no_grad():
        model.gate_logits.fill_(-4.0)
        model.gate_logits[0, 3] = 5.0
        model.gate_logits[1, 11] = 6.0
    atoms = torch.randn(3, 5, model.atom_count, dtype=torch.float64)

    hard_st = model(atoms, tau=0.1, mode="hard_st")
    receipt = harden_checkpoint(model.checkpoint_logits(), model.atom_manifest, 0.1)
    replay = evaluate_hard_ast_from_atoms(receipt.ast, model.atom_manifest, atoms)

    assert receipt.ast.op == "Sub"
    assert receipt.positive.atom_index == 3
    assert receipt.negative.atom_index == 11
    torch.testing.assert_close(hard_st, replay, rtol=0.0, atol=0.0)
    hard_st.sum().backward()
    assert model.gate_logits.grad is not None
    assert torch.all(model.gate_logits.grad.abs().sum(dim=1) > 0)


def test_hardener_is_checkpoint_only_and_uses_hash_ties_without_cancellation() -> None:
    from mirage_kan.mining.e3 import build_profile_atom_bank, harden_checkpoint

    forbidden = ("target", "validation", "label", "raw", "outcome", "score")
    parameters = inspect.signature(harden_checkpoint).parameters
    assert tuple(parameters) == ("logits", "atom_manifest", "tau")
    assert not any(token in name for name in parameters for token in forbidden)

    manifest = build_profile_atom_bank("reversal")
    tied = torch.zeros(2, len(manifest), dtype=torch.float64)
    first = harden_checkpoint(tied, manifest, 0.1)
    second = harden_checkpoint(tied.clone(), manifest, 0.1)
    ordered = sorted(manifest, key=lambda atom: atom.canonical_hash)

    assert first == second
    assert first.ast.identity == second.ast.identity
    assert first.positive.canonical_hash == ordered[0].canonical_hash
    assert first.negative.canonical_hash == ordered[1].canonical_hash
    assert first.positive.atom_index != first.negative.atom_index
    assert first.rejected_alternates
    assert any(
        item.reason == "same_atom_cancellation" for item in first.rejected_alternates
    )


def test_temperature_mode_objective_and_fidelity_helpers_are_frozen() -> None:
    from mirage_kan.mining.e3 import (
        CategoricalE3KAN,
        forward_mode_at_step,
        soft_hard_fidelity,
        temperature_at_step,
        training_objective,
    )

    assert temperature_at_step(0) == 2.0
    assert temperature_at_step(299) == pytest.approx(0.1)
    assert forward_mode_at_step(224) == "soft"
    assert forward_mode_at_step(225) == "hard_st"
    assert forward_mode_at_step(299) == "hard_st"

    model = CategoricalE3KAN("long_price", seed=731001)
    atoms = torch.randn(4, 6, model.atom_count, dtype=torch.float64)
    target = atoms[..., 0] - atoms[..., 1]
    terms = training_objective(model, atoms, target, tau=1.0, mode="soft")
    expected = (
        -terms["mean_daily_ic"]
        + 0.001 * terms["entropy"]
        + 0.01 * terms["edge_overlap"]
    )
    torch.testing.assert_close(terms["total_loss"], expected)

    same = soft_hard_fidelity(target, target.clone())
    assert same == {"pearson": 1.0, "nrmse": 0.0, "max_absolute_error": 0.0}


@pytest.mark.parametrize("profile", ["trend", "", "SHORT_PRICE"])
def test_illegal_profile_fails(profile: str) -> None:
    from mirage_kan.mining.e3 import CategoricalE3KAN, build_profile_atom_bank

    with pytest.raises(ValueError, match="profile"):
        build_profile_atom_bank(profile)
    with pytest.raises(ValueError, match="profile"):
        CategoricalE3KAN(profile, seed=1)


def test_illegal_shapes_tau_modes_and_nonfinite_values_fail() -> None:
    from mirage_kan.mining.e3 import (
        CategoricalE3KAN,
        evaluate_hard_ast_from_atoms,
        forward_mode_at_step,
        harden_checkpoint,
        temperature_at_step,
        training_objective,
    )

    model = CategoricalE3KAN("short_price", seed=730001)
    valid = torch.randn(2, 3, model.atom_count, dtype=torch.float64)
    target = torch.randn(2, 3, dtype=torch.float64)
    logits = model.checkpoint_logits()

    with pytest.raises(ValueError, match="shape"):
        model(valid[..., :-1], tau=1.0, mode="soft")
    with pytest.raises(ValueError, match="tau"):
        model(valid, tau=0.0, mode="soft")
    with pytest.raises(ValueError, match="mode"):
        model(valid, tau=1.0, mode="hard")
    with pytest.raises(ValueError, match="shape"):
        training_objective(model, valid, target[:, :-1], tau=1.0, mode="soft")
    with pytest.raises(ValueError, match="finite"):
        training_objective(
            model,
            valid,
            target.masked_fill(
                torch.zeros_like(target, dtype=torch.bool).index_fill(
                    1, torch.tensor([0]), True
                ),
                float("nan"),
            ),
            tau=1.0,
            mode="soft",
        )
    with pytest.raises(ValueError, match="shape"):
        harden_checkpoint(logits[:, :-1], model.atom_manifest, 0.1)
    with pytest.raises(ValueError, match="tau"):
        harden_checkpoint(logits, model.atom_manifest, -0.1)

    receipt = harden_checkpoint(logits, model.atom_manifest, 0.1)
    with pytest.raises(ValueError, match="shape"):
        evaluate_hard_ast_from_atoms(receipt.ast, model.atom_manifest, valid[..., :-1])
    with pytest.raises(ValueError, match="update_index"):
        temperature_at_step(300)
    with pytest.raises(ValueError, match="update_index"):
        forward_mode_at_step(-1)
