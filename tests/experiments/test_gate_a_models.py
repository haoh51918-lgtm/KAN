from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def test_cubic_basis_matches_worked_knot_literal() -> None:
    from mirage_kan.experiments.gate_a.models import cubic_bspline_basis

    x = torch.tensor([0.0], dtype=torch.float64)
    knots = torch.arange(-4.0, 5.0, dtype=torch.float64)
    basis = cubic_bspline_basis(x, knots)
    torch.testing.assert_close(
        basis,
        torch.tensor([[0.0, 1 / 6, 2 / 3, 1 / 6, 0.0]], dtype=torch.float64),
        rtol=1e-14,
        atol=1e-14,
    )
    torch.testing.assert_close(basis.sum(dim=-1), torch.ones(1, dtype=torch.float64))


def test_e1_is_additive_cubic_spline_and_exposes_edges() -> None:
    from mirage_kan.experiments.gate_a.models import FreeSplineKAN

    model = FreeSplineKAN(input_count=2, grid_intervals=2, grid_range=(-1.0, 1.0))
    model = model.to(dtype=torch.float64)
    with torch.no_grad():
        model.coefficients[0].fill_(1.0)
        model.coefficients[1].fill_(2.0)
        model.output_bias.zero_()
    inputs = torch.tensor([[0.0, 0.0], [0.5, -0.5]], dtype=torch.float64)
    edges = model.edge_outputs(inputs)
    torch.testing.assert_close(
        edges, torch.tensor([[1.0, 2.0], [1.0, 2.0]], dtype=torch.float64)
    )
    torch.testing.assert_close(model(inputs), edges.sum(dim=1, keepdim=True))
    curves = model.edge_curves(torch.linspace(-1, 1, 9, dtype=torch.float64))
    assert curves.shape == (9, 2)
    assert model.parameter_count == 11
    assert model.spline_degree == 3


def test_capacity_spec_and_c6_are_parameter_matched() -> None:
    from mirage_kan.experiments.gate_a.models import CAPACITY_SPEC, MatchedMLP

    manifest = json.loads(
        Path("configs/model_specs/s1_gate_a_capacity_v0.json").read_text()
    )
    assert manifest["e3_e4_shared"]["e4_declared_trainable_parameters"] == 361
    assert CAPACITY_SPEC.e4_parameter_count == 361
    model = MatchedMLP(CAPACITY_SPEC)
    assert model.parameter_count == 361
    assert model.hidden_width == 45
    assert model.parameter_match_relative_gap <= 0.10
    prediction = model(torch.zeros(3, 6, dtype=torch.float64))
    assert prediction.shape == (3, 1)
    assert prediction.dtype == torch.float64
    same = MatchedMLP(CAPACITY_SPEC)
    for left, right in zip(model.parameters(), same.parameters(), strict=True):
        torch.testing.assert_close(left, right)


def test_e1_supports_available_cuda_float64() -> None:
    if not torch.cuda.is_available():
        return
    from mirage_kan.experiments.gate_a.models import FreeSplineKAN

    model = FreeSplineKAN(2, grid_intervals=2).cuda().double()
    result = model(torch.zeros(4, 2, dtype=torch.float64, device="cuda"))
    assert result.is_cuda
    assert result.dtype == torch.float64
    assert np.isfinite(result.detach().cpu().numpy()).all()
