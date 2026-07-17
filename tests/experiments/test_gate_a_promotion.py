from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch


def test_promotion_families_match_worked_equations_and_round_trip() -> None:
    from mirage_kan.experiments.gate_a.promotion import (
        PromotedPrimitive,
        evaluate_promotion_family,
    )

    x = np.array([-1.0, 0.0, 2.0])
    params = np.array([2.0, 0.5, 3.0, 0.25])
    spec = json.loads(
        Path("configs/model_specs/s1_gate_a_promotion_v0.json").read_text()
    )
    assert spec["governance"]["duplicate_affine_fit_nrmse_maximum"] == 0.05
    assert spec["governance"]["non_duplication_fit"]["semantics"] == (
        "full_initial_primitive_input_and_output_affine_transform"
    )
    expected_exp = np.array(
        [-2.0 * (1.0 - np.exp(-0.5)), 0.0, 3.0 * (1.0 - np.exp(-0.5))]
    )
    np.testing.assert_allclose(
        evaluate_promotion_family("asymmetric_exponential_saturation_v1", x, params),
        expected_exp,
    )
    np.testing.assert_allclose(
        evaluate_promotion_family("asymmetric_rational_saturation_v1", x, params),
        [-2 / 1.5, 0.0, 6 / 1.5],
    )
    np.testing.assert_allclose(
        evaluate_promotion_family("two_sided_tanh_v1", x, params),
        [-2 * np.tanh(0.5), 0.0, 3 * np.tanh(0.5)],
    )
    primitive = PromotedPrimitive.create(
        "asymmetric_exponential_saturation_v1", params
    )
    restored = PromotedPrimitive.from_canonical(primitive.canonical_serialization())
    assert restored.primitive_id == primitive.primitive_id
    np.testing.assert_array_equal(restored.evaluate(x), primitive.evaluate(x))


def test_eligibility_centers_and_scales_but_never_flips_sign() -> None:
    from mirage_kan.experiments.gate_a.promotion import (
        ResidualShape,
        assess_residual_eligibility,
    )

    z = np.linspace(-4, 4, 801)
    shape = np.where(z < 0, -(1 - np.exp(z)), 1.5 * (1 - np.exp(-0.5 * z)))
    shapes = (
        ResidualShape(11, "Return(Close,5)", 0.7, z, shape),
        ResidualShape(12, "Return(Close,5)", 0.8, z, 3.0 * shape + 9.0),
        ResidualShape(13, "Return(Close,5)", 0.9, z, -shape),
    )
    result = assess_residual_eligibility(shapes)
    assert result.eligible is True
    assert result.eligible_seeds == (11, 12)
    assert result.correlations["11:12"] >= 0.999999
    assert result.correlations["11:13"] <= -0.999999
    assert result.sign_flip_allowed is False
    with pytest.raises(ValueError, match="frozen 801-point grid"):
        ResidualShape(14, "Return(Close,5)", 0.5, z + 1e-12, shape)
    with pytest.raises(ValueError, match="read-only"):
        shapes[0].residual_values[0] = 999.0


def test_governed_promotion_fits_continuously_and_rejects_dictionary_duplicate() -> None:
    from mirage_kan.experiments.gate_a.promotion import (
        ResidualShape,
        assess_residual_eligibility,
        fit_governed_promotion,
    )

    assert "test" not in inspect.signature(fit_governed_promotion).parameters
    assert "target" not in inspect.signature(fit_governed_promotion).parameters
    z = np.linspace(-4, 4, 801)
    asymmetric = np.where(
        z < 0,
        -0.6 * (1 - np.exp(2.5 * z)),
        1.8 * (1 - np.exp(-0.25 * z)),
    )
    shapes = tuple(
        ResidualShape(seed, "Return(Close,5)", 0.75, z, asymmetric)
        for seed in (101, 102)
    )
    promoted = fit_governed_promotion(shapes, assess_residual_eligibility(shapes))
    assert promoted.promoted is True
    assert promoted.family_id == "asymmetric_exponential_saturation_v1"
    assert promoted.primitive_id.startswith("asymmetric_exponential_saturation_v1_")
    assert promoted.governance["all_passed"] is True
    assert all(fit["continuous_optimization"] for fit in promoted.seed_fits.values())
    exponential_audit = next(
        audit
        for audit in promoted.candidate_audits
        if audit["family_id"] == "asymmetric_exponential_saturation_v1"
    )
    assert exponential_audit["checks"]["non_duplication"] is True
    assert exponential_audit["duplicate_audit"]["primitive"] == "Tanh"

    duplicate = np.tanh(z)
    duplicate_shapes = tuple(
        ResidualShape(seed, "Return(Close,5)", 0.8, z, duplicate)
        for seed in (201, 202)
    )
    rejected = fit_governed_promotion(
        duplicate_shapes, assess_residual_eligibility(duplicate_shapes)
    )
    assert rejected.promoted is False
    assert rejected.status == "no_governed_candidate"
    assert any(
        not audit["checks"]["non_duplication"]
        for audit in rejected.candidate_audits
    )
    transformed_tanh = np.tanh(2.0 * z)
    transformed_shapes = tuple(
        ResidualShape(seed, "Return(Close,5)", 0.8, z, transformed_tanh)
        for seed in (211, 212)
    )
    transformed_rejected = fit_governed_promotion(
        transformed_shapes, assess_residual_eligibility(transformed_shapes)
    )
    assert transformed_rejected.promoted is False
    tanh_audit = next(
        audit
        for audit in transformed_rejected.candidate_audits
        if audit["family_id"] == "two_sided_tanh_v1"
    )
    assert tanh_audit["checks"]["non_duplication"] is False
    assert tanh_audit["duplicate_audit"]["primitive"] == "Tanh"
    assert tanh_audit["duplicate_audit"]["estimated_flops"] > 0

    nonparametric = np.sin(3.0 * z) + 0.2 * np.sin(11.0 * z)
    poor_shapes = tuple(
        ResidualShape(seed, "Return(Close,5)", 0.8, z, nonparametric)
        for seed in (301, 302)
    )
    poor = fit_governed_promotion(
        poor_shapes, assess_residual_eligibility(poor_shapes)
    )
    assert poor.promoted is False
    assert all(
        not audit["checks"]["low_complexity_approximation"]
        for audit in poor.candidate_audits
    )


def test_hard_refit_changes_only_output_affines_and_reloads_independently(tmp_path) -> None:
    from mirage_kan.experiments.gate_a.promotion import (
        PromotedPrimitive,
        load_promoted_hard_model,
        refit_promoted_hard_model,
        save_promoted_hard_model,
    )
    from mirage_kan.experiments.gate_a.symbolic import HardAnalyticalKAN

    hard = HardAnalyticalKAN(
        primitive_indices=torch.tensor([0, 4]),
        affine_parameters=torch.tensor(
            [[1.5, 0.2, 0.7, -0.1], [0.8, -0.3, 1.2, 0.4]],
            dtype=torch.float64,
        ),
        output_bias=torch.tensor([5.0], dtype=torch.float64),
        source_names=("Return(Close,2)", "Return(Close,5)"),
    )
    primitive = PromotedPrimitive.create(
        "asymmetric_exponential_saturation_v1", [0.8, 1.5, 1.4, 0.7]
    )
    rng = np.random.default_rng(77)
    train = rng.normal(size=(80, 2))
    validation = rng.normal(size=(40, 2))

    def truth(x: np.ndarray) -> np.ndarray:
        edge = hard.edge_outputs(torch.as_tensor(x)).numpy()
        return 2.0 * edge[:, 0] - edge[:, 1] + 0.6 * primitive.evaluate(x[:, 1]) + 0.4

    model, evidence = refit_promoted_hard_model(
        hard,
        primitive,
        source_index=1,
        train_features=train,
        train_noisy_target=truth(train),
        validation_features=validation,
        validation_clean_truth=truth(validation),
    )
    np.testing.assert_allclose(model.evaluate(validation), truth(validation), atol=1e-12)
    source = model.selected_source_metadata(torch.as_tensor(validation))
    assert source["source"] == "Return(Close,5)"
    assert source["window"] == 5
    assert 0.0 <= source["selected_input_mass"] <= 1.0
    np.testing.assert_array_equal(
        model.analytical_affine_parameters.numpy(), hard.affine_parameters.numpy()
    )
    assert evidence["discrete_reselection"] is False
    assert evidence["test_access"] is False
    assert evidence["all_free_spline_paths_closed_by_hardening"] is True
    manifest_path = save_promoted_hard_model(model, tmp_path / "hard", evidence=evidence)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["publication"]["manifest_last"] is True
    restored = load_promoted_hard_model(manifest_path)
    np.testing.assert_array_equal(restored.evaluate(validation), model.evaluate(validation))
    with pytest.raises(FileExistsError):
        save_promoted_hard_model(model, tmp_path / "hard", evidence=evidence)
