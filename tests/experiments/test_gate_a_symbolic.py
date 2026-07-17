from __future__ import annotations

import json
from pathlib import Path
import tomllib

import pytest
import torch


def test_frozen_primitives_match_independently_worked_literals() -> None:
    from mirage_kan.experiments.gate_a.symbolic import (
        PRIMITIVE_NAMES,
        evaluate_primitive,
    )

    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=torch.float64)
    expected = {
        "Identity": x,
        "Abs": torch.tensor([2.0, 0.5, 0.0, 0.5, 2.0]),
        "Square": torch.tensor([4.0, 0.25, 0.0, 0.25, 4.0]),
        "SignedLog1p": torch.sign(x) * torch.log1p(torch.abs(x)),
        "Tanh": torch.tanh(x),
        "Clip(-1,1)": torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0]),
        "PositiveHinge(0)": torch.tensor([0.0, 0.0, 0.0, 0.5, 2.0]),
        "NegativeHinge(0)": torch.tensor([2.0, 0.5, 0.0, 0.0, 0.0]),
    }
    assert PRIMITIVE_NAMES == tuple(expected)
    assert "Exp" not in PRIMITIVE_NAMES
    for name, literal in expected.items():
        torch.testing.assert_close(
            evaluate_primitive(name, x), literal.to(dtype=torch.float64)
        )
    extreme = torch.tensor([-torch.inf, torch.inf, torch.nan], dtype=torch.float64)
    for name in PRIMITIVE_NAMES:
        with pytest.raises(ValueError, match="finite inputs"):
            evaluate_primitive(name, extreme)
        assert torch.isfinite(
            evaluate_primitive(name, torch.tensor([1e150], dtype=torch.float64))
        ).all()
    with pytest.raises(FloatingPointError, match="Square"):
        evaluate_primitive(
            "Square", torch.tensor([torch.finfo(torch.float64).max], dtype=torch.float64)
        )


def test_hard_model_is_independent_and_has_canonical_description() -> None:
    from mirage_kan.experiments.gate_a.symbolic import HardAnalyticalKAN

    model = HardAnalyticalKAN(
        primitive_indices=torch.tensor([0, 5]),
        affine_parameters=torch.tensor(
            [[2.0, 1.0, 3.0, -1.0], [1.0, 0.0, 2.0, 0.5]],
            dtype=torch.float64,
        ),
        output_bias=torch.tensor([0.25], dtype=torch.float64),
        source_names=("Return(Close,2)", "Return(Close,5)"),
    )
    values = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    # 3 * Identity(2*1+1)-1 + 2*Clip(2)+0.5 + output bias.
    torch.testing.assert_close(
        model(values), torch.tensor([[10.75]], dtype=torch.float64)
    )
    assert not tuple(model.parameters())
    description = model.canonical_description()
    assert description["type"] == "AdditiveSymbolicKAN"
    assert description["edges"][1]["primitive"] == "Clip(-1,1)"
    serialized = model.canonical_serialization()
    assert serialized == model.canonical_serialization()
    assert model.description_length == len(serialized.encode("utf-8"))


def test_shared_fidelity_and_zero_residual_energy_semantics() -> None:
    from mirage_kan.experiments.gate_a.symbolic import (
        fidelity_metrics,
        residual_energy_ratio,
    )

    soft = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    hard = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    assert fidelity_metrics(soft, hard) == {
        "pearson": 1.0,
        "nrmse": 0.0,
        "max_absolute_error": 0.0,
    }
    zeros = torch.zeros(4, 2, dtype=torch.float64)
    torch.testing.assert_close(
        residual_energy_ratio(zeros, zeros), torch.zeros(2, dtype=torch.float64)
    )
    residual = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
    ratio = residual_energy_ratio(residual, torch.zeros_like(residual))
    assert torch.isfinite(ratio).all()
    torch.testing.assert_close(ratio, torch.tensor([1e12], dtype=torch.float64))


def test_symbolic_choices_are_prospectively_frozen() -> None:
    spec = json.loads(
        Path("configs/model_specs/s1_gate_a_symbolic_v0.json").read_text()
    )
    assert spec["decision_timing"] == "before_any_e2_e3_e4_smoke_metric"
    assert spec["initial_primitives"] == [
        "Identity",
        "Abs",
        "Square",
        "SignedLog1p",
        "Tanh",
        "Clip(-1,1)",
        "PositiveHinge(0)",
        "NegativeHinge(0)",
    ]
    assert spec["hardening"]["tie_break"] == "primitive_dictionary_order"
    families = {item["id"] for item in spec["promotion"]["candidate_families"]}
    assert families == {
        "asymmetric_exponential_saturation_v1",
        "asymmetric_rational_saturation_v1",
        "two_sided_tanh_v1",
    }


def test_gate_a_runtime_dependencies_are_declared() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    dependencies = project["dependencies"]
    assert any(item.startswith("scipy>=") for item in dependencies)
    assert any(item.startswith("torch>=") for item in dependencies)


def test_e3_has_real_gates_schedule_accounting_and_deterministic_hardening() -> None:
    from mirage_kan.experiments.gate_a.models import SymbolicKAN

    model = SymbolicKAN(input_count=2)
    assert model.parameter_count == 81
    inputs = torch.tensor([[0.25, -0.5], [1.0, 2.0]], dtype=torch.float64)
    gates = model.gate_outputs(temperature=2.0)
    assert gates.shape == (2, 8)
    torch.testing.assert_close(gates.sum(dim=1), torch.ones(2, dtype=torch.float64))
    assert model.temperature(step=0, max_steps=10) == 2.0
    assert model.temperature(step=10, max_steps=10) == pytest.approx(0.1)
    terms = model.regularization_terms(inputs, temperature=2.0)
    assert set(terms) == {
        "gate_entropy",
        "gate_sparsity",
        "spline_residual_raw_energy",
        "spline_residual_ratio",
    }
    assert float(terms["gate_entropy"].detach()) > 0
    assert float(terms["gate_sparsity"].detach()) > 0
    assert float(terms["spline_residual_raw_energy"].detach()) == 0
    assert float(terms["spline_residual_ratio"].detach()) == 0

    with torch.no_grad():
        model.gate_logits.zero_()
    hard = model.harden()
    assert hard.primitive_indices.tolist() == [0, 0]
    before = hard(inputs).clone()
    with torch.no_grad():
        model.gate_logits[:, 7] = 100.0
        model.affine_parameters.add_(2.0)
    torch.testing.assert_close(hard(inputs), before)


def test_e4_is_e3_plus_actual_cubic_residual_and_has_361_parameters() -> None:
    from mirage_kan.experiments.gate_a.models import (
        CAPACITY_SPEC,
        SymbolicKAN,
        SymbolicResidualKAN,
    )

    e3 = SymbolicKAN()
    e4 = SymbolicResidualKAN()
    assert e3.parameter_count == 241
    assert e4.parameter_count == CAPACITY_SPEC.e4_parameter_count == 361
    assert e4.spline_degree == 3
    assert e4.coefficients.shape == (6, 19)
    assert e4.residual_scales.shape == (6,)
    torch.testing.assert_close(e3.gate_logits, e4.gate_logits)
    torch.testing.assert_close(e3.affine_parameters, e4.affine_parameters)

    inputs = torch.zeros(3, 6, dtype=torch.float64)
    torch.testing.assert_close(e3.analytical_edge_outputs(inputs), e4.analytical_edge_outputs(inputs))
    with torch.no_grad():
        e4.coefficients.fill_(1.0)
        e4.residual_scales.fill_(2.0)
    residual = e4.residual_edge_outputs(inputs)
    torch.testing.assert_close(residual, torch.full((3, 6), 2.0, dtype=torch.float64))
    torch.testing.assert_close(
        e4.edge_outputs(inputs), e4.analytical_edge_outputs(inputs) + residual
    )
    assert torch.all(e4.residual_energy(inputs) > 0)


def test_e4_penalty_uses_rho_not_raw_residual_mse() -> None:
    from mirage_kan.experiments.gate_a.models import SymbolicResidualKAN

    model = SymbolicResidualKAN(input_count=2)
    inputs = torch.zeros(3, 2, dtype=torch.float64)
    with torch.no_grad():
        model.coefficients.fill_(1.0)
        model.residual_scales.fill_(2.0)
    terms = model.regularization_terms(inputs, temperature=1.0)
    torch.testing.assert_close(
        terms["spline_residual_raw_energy"],
        torch.tensor(4.0, dtype=torch.float64),
    )
    torch.testing.assert_close(
        terms["spline_residual_ratio"],
        torch.tensor(1.0, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    penalty, accounting = model.training_penalty(inputs, step=1, max_steps=1)
    expected = (
        model.settings.entropy_weight * terms["gate_entropy"]
        + model.settings.sparsity_weight * terms["gate_sparsity"]
        + model.settings.spline_residual_energy_weight
        * terms["spline_residual_ratio"]
    )
    torch.testing.assert_close(penalty, expected)
    assert accounting["spline_residual_raw_energy"] == pytest.approx(4.0)
    assert accounting["spline_residual_ratio"] == pytest.approx(1.0)


def test_source_mass_is_centered_bias_invariant_and_explicit_when_zero() -> None:
    from mirage_kan.experiments.gate_a.models import SymbolicKAN

    model = SymbolicKAN(input_count=2)
    inputs = torch.tensor(
        [[-2.0, -1.0], [-1.0, 0.0], [1.0, 2.0], [2.0, 4.0]],
        dtype=torch.float64,
    )
    before = model.source_mass(inputs)
    with torch.no_grad():
        model.affine_parameters[..., 3] += torch.tensor(
            [[100.0], [-250.0]], dtype=torch.float64
        )
    torch.testing.assert_close(model.source_mass(inputs), before)
    torch.testing.assert_close(before.sum(), torch.tensor(1.0, dtype=torch.float64))

    with torch.no_grad():
        model.affine_parameters[..., 2].zero_()
    zero_mass = model.source_mass(inputs)
    torch.testing.assert_close(zero_mass, torch.zeros(2, dtype=torch.float64))
    selected = model.selected_source_metadata(inputs)
    assert selected == {
        "edge": None,
        "primitive": None,
        "source": None,
        "window": None,
        "selected_input_mass": 0.0,
        "selection_status": "zero_contribution_energy",
    }


def test_shape_interface_uses_sealed_801_point_grid() -> None:
    from mirage_kan.experiments.gate_a.models import SymbolicResidualKAN

    report = SymbolicResidualKAN().shape_report()
    assert report.grid.shape == (801,)
    assert report.grid[0] == -4
    assert report.grid[-1] == 4
    assert report.soft_edges.shape == (801, 6)
    assert report.hard_edges.shape == (801, 6)
    assert report.residual_edges is not None
    assert report.residual_edges.shape == (801, 6)


def test_e2_fits_each_e1_edge_to_exactly_eight_train_only_families() -> None:
    import inspect

    from torch import nn

    from mirage_kan.experiments.gate_a.posthoc import symbolify_e1

    class FittedE1Literal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.output_bias = nn.Parameter(torch.tensor([0.75], dtype=torch.float64))
            self.register_buffer("edge_bias", torch.zeros(2, dtype=torch.float64))

        def edge_outputs(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs + self.edge_bias

    assert "target" not in inspect.signature(symbolify_e1).parameters
    train_features = torch.tensor(
        [[-2.0, -1.0], [-1.0, -0.5], [0.5, 1.0], [2.0, 3.0]],
        dtype=torch.float64,
    )
    result = symbolify_e1(
        FittedE1Literal(),
        train_features,
        source_names=("Return(Close,2)", "Return(Close,5)"),
    )
    assert result.fit_count == 16
    assert len(result.all_fits) == 16
    assert [fit.primitive for fit in result.selected_fits] == ["Identity", "Identity"]
    assert result.fit_manifest["fits_per_edge"] == 8
    assert result.fit_manifest["selection_data"] == "train_inputs_and_e1_edges_only"
    torch.testing.assert_close(
        result.hard_model(train_features),
        train_features.sum(dim=1, keepdim=True) + 0.75,
        atol=1e-8,
        rtol=1e-8,
    )
    assert result.hard_model.output_bias.item() == 0.75
    assert result.hard_model.selected_metadata()[1]["window"] == 5
    before_mass = result.source_mass(train_features)
    with torch.no_grad():
        result.source_model.edge_bias += torch.tensor([100.0, -250.0])
    torch.testing.assert_close(result.source_mass(train_features), before_mass)


def test_e2_can_load_a_selected_e1_checkpoint_without_retraining(tmp_path) -> None:
    from mirage_kan.experiments.gate_a.models import FreeSplineKAN
    from mirage_kan.experiments.gate_a.posthoc import symbolify_e1_checkpoint

    selected = FreeSplineKAN(input_count=1, grid_intervals=2)
    with torch.no_grad():
        selected.coefficients.fill_(0.25)
        selected.output_bias.fill_(1.5)
    checkpoint = tmp_path / "selected.pt"
    torch.save({"model_state_dict": selected.state_dict()}, checkpoint)
    fresh = FreeSplineKAN(input_count=1, grid_intervals=2)
    result = symbolify_e1_checkpoint(
        checkpoint,
        fresh,
        torch.tensor([[-0.5], [0.0], [0.5]], dtype=torch.float64),
        source_names=("Return(Close,2)",),
    )
    assert result.fit_count == 8
    assert fresh.output_bias.item() == 1.5
    assert result.hard_model.output_bias.item() == 1.5


def test_hard_export_is_manifest_last_and_no_replace(tmp_path) -> None:
    from mirage_kan.experiments.gate_a.symbolic import (
        HardAnalyticalKAN,
        load_hard_export,
        save_hard_export,
    )

    hard = HardAnalyticalKAN(
        primitive_indices=torch.tensor([0]),
        affine_parameters=torch.tensor([[1.0, 0.0, 1.0, 0.0]]),
        output_bias=torch.tensor([0.0]),
        source_names=("Return(Close,5)",),
    )
    manifest_path = save_hard_export(hard, tmp_path / "export", arm="E3")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["arm"] == "E3"
    assert Path(manifest["paths"]["checkpoint"]).is_file()
    assert manifest["canonical"] == hard.canonical_description()
    restored = load_hard_export(manifest_path)
    values = torch.tensor([[0.25]], dtype=torch.float64)
    torch.testing.assert_close(restored(values), hard(values))
    with pytest.raises(FileExistsError):
        save_hard_export(hard, tmp_path / "export", arm="E3")
