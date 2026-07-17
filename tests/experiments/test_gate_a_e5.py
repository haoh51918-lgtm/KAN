from __future__ import annotations

import json
import inspect
from pathlib import Path

import numpy as np
import pytest


def test_e5_choices_are_frozen_and_executable_ast_round_trips() -> None:
    from mirage_kan.experiments.gate_a.e5 import (
        E5Atom,
        E5ExecutableModel,
        E5Structure,
    )

    spec = json.loads(Path("configs/model_specs/s1_gate_a_e5_v0.json").read_text())
    assert spec["decision_timing"] == "before_any_e5_smoke_metric"
    assert spec["budget"] == {
        "max_distinct_valid_full_ast_evaluations": 12000,
        "max_ast_depth": 5,
        "max_ast_nodes": 15,
        "duplicate_policy": "record_attempt_without_consuming_full_fit_evaluation",
        "invalid_policy": "record_separately_without_consuming_valid_evaluation",
        "budget_exhaustion_policy": (
            "record_unfitted_unique_attempt_and_stop_automatic_generation"
        ),
    }
    forbidden = set(spec["forbidden"])
    assert {"Exp", "AsymmetricSaturation", "PiecewiseExponential"} <= forbidden
    structure = E5Structure(
        (
            E5Atom(0, "Return(Close,2)", "Identity"),
            E5Atom(1, "Return(Close,5)", "Square"),
        )
    )
    model = E5ExecutableModel(
        structure=structure,
        coefficients=np.array([2.0, -0.5], dtype=np.float64),
        intercept=1.0,
    )
    features = np.array(
        [[1.0, 2.0], [-1.0, 3.0], [0.5, -2.0]], dtype=np.float64
    )
    # 1 + 2*Identity(x0) - 0.5*Square(x1), independently worked.
    np.testing.assert_allclose(model.evaluate(features), [1.0, -5.5, 0.0])
    assert model.complexity() == {
        "ast_node_count": 10,
        "ast_depth": 4,
        "free_constants": 3,
        "serialized_description_length": len(
            model.canonical_serialization().encode("utf-8")
        ),
    }
    payload = json.loads(model.canonical_serialization())
    assert payload["op"] == "Add"
    assert E5ExecutableModel.from_canonical_serialization(
        model.canonical_serialization()
    ).canonical_serialization() == model.canonical_serialization()
    with pytest.raises(ValueError, match="read-only"):
        model.coefficients[0] = 99.0
    assert model.source_metadata() == [
        {"source_index": 0, "source": "Return(Close,2)", "window": 2},
        {"source_index": 1, "source": "Return(Close,5)", "window": 5},
    ]
    with pytest.raises(ValueError, match="non-finite"):
        model.evaluate(np.array([[np.nan, 1.0]], dtype=np.float64))
    with pytest.raises(ValueError, match="referenced source columns"):
        model.evaluate(np.ones((2, 1), dtype=np.float64))
    with pytest.raises(ValueError, match="at least one sample"):
        model.source_mass(np.empty((0, 2), dtype=np.float64))
    three_terms = E5Structure(
        structure.atoms
        + (E5Atom(2, "Return(Close,10)", "PositiveHinge(0)"),)
    )
    assert three_terms.ast_node_count == 14
    assert three_terms.ast_depth == 4


def test_e5_search_recovers_a_worked_two_source_additive_expression() -> None:
    from mirage_kan.experiments.gate_a.data import FEATURE_NAMES
    from mirage_kan.experiments.gate_a.e5 import E5SearchSettings, search_e5

    rng = np.random.default_rng(20260716)
    train = rng.normal(size=(96, 6))
    validation = rng.normal(size=(48, 6))
    train_noisy = 0.75 + 1.5 * train[:, 0] - 0.8 * np.square(train[:, 1])
    validation_clean = (
        0.75 + 1.5 * validation[:, 0] - 0.8 * np.square(validation[:, 1])
    )
    assert "test" not in inspect.signature(search_e5).parameters
    result = search_e5(
        train,
        train_noisy,
        validation,
        validation_clean,
        source_names=FEATURE_NAMES,
        settings=E5SearchSettings(max_distinct_valid_evaluations=80, seed=8675309),
    )
    selected = result.selected_model
    assert [
        (atom.source_index, atom.primitive)
        for atom in selected.structure.canonical_atoms()
    ] == [(0, "Identity"), (1, "Square")]
    np.testing.assert_allclose(selected.coefficients, [1.5, -0.8], atol=1e-12)
    assert selected.intercept == pytest.approx(0.75, abs=1e-12)
    assert result.selected_validation_nrmse < 1e-12
    assert result.accounting["distinct_valid_ast_evaluations"] == 80
    assert result.accounting["fit_attempts"] == 80
    assert result.accounting["successful_fits"] == 80
    assert result.accounting["budget_exhausted"] is True
    assert result.generation_mode == "deterministic_frozen_generator"


def test_e5_duplicate_invalid_and_budget_ledgers_are_disjoint() -> None:
    from mirage_kan.experiments.gate_a.data import FEATURE_NAMES
    from mirage_kan.experiments.gate_a.e5 import (
        E5Atom,
        E5SearchSettings,
        E5Structure,
        search_e5,
    )

    train = np.arange(48, dtype=np.float64).reshape(8, 6) / 10
    validation = np.arange(24, dtype=np.float64).reshape(4, 6) / 10
    train[:, 2] = np.nan
    validation[:, 2] = np.nan
    train[:, 4] = np.finfo(np.float64).max
    validation[:, 4] = np.finfo(np.float64).max
    atom0 = E5Atom(0, FEATURE_NAMES[0], "Identity")
    atom1 = E5Atom(1, FEATURE_NAMES[1], "Identity")
    atom2 = E5Atom(2, FEATURE_NAMES[2], "Identity")
    atom3 = E5Atom(3, FEATURE_NAMES[3], "Identity")
    overflowing_atom = E5Atom(4, FEATURE_NAMES[4], "Square")
    proposals = (
        E5Structure((atom0,)),
        E5Structure((atom0, atom0)),
        E5Structure((E5Atom(0, FEATURE_NAMES[0], "Exp"),)),
        E5Structure((atom2,)),
        E5Structure((overflowing_atom,)),
        E5Structure((atom1,)),
        E5Structure((atom3,)),
    )
    result = search_e5(
        train,
        train[:, 0],
        validation,
        validation[:, 0],
        source_names=FEATURE_NAMES,
        settings=E5SearchSettings(max_distinct_valid_evaluations=2),
        candidate_proposals=proposals,
    )
    assert [entry["status"] for entry in result.ledger] == [
        "evaluated",
        "duplicate",
        "invalid_ast",
        "invalid_execution",
        "invalid_execution",
        "evaluated",
        "budget_exhausted",
    ]
    expected_counts = {
        "attempted_candidates": 7,
        "distinct_valid_ast_evaluations": 2,
        "duplicate_attempts": 1,
        "invalid_ast_attempts": 1,
        "invalid_execution_attempts": 2,
        "fit_attempts": 2,
        "successful_fits": 2,
        "fit_failures": 0,
        "budget_exhausted_attempts": 1,
        "budget_exhausted": True,
    }
    for name, expected in expected_counts.items():
        assert result.accounting[name] == expected
    assert result.accounting["wall_clock_seconds"] >= 0
    replay = search_e5(
        train,
        train[:, 0],
        validation,
        validation[:, 0],
        source_names=FEATURE_NAMES,
        settings=E5SearchSettings(max_distinct_valid_evaluations=2),
        candidate_proposals=proposals,
    )
    assert replay.ledger == result.ledger
    assert (
        replay.selected_model.canonical_serialization()
        == result.selected_model.canonical_serialization()
    )


def test_e5_selection_prefers_lower_complexity_within_frozen_tolerance() -> None:
    from mirage_kan.experiments.gate_a.data import FEATURE_NAMES
    from mirage_kan.experiments.gate_a.e5 import (
        E5Atom,
        E5SearchSettings,
        E5Structure,
        search_e5,
    )

    rng = np.random.default_rng(1234)
    train = rng.normal(size=(128, 6))
    validation = rng.normal(size=(64, 6))
    train_target = train[:, 0] + 0.002 * train[:, 1]
    validation_truth = validation[:, 0] + 0.002 * validation[:, 1]
    first = E5Atom(0, FEATURE_NAMES[0], "Identity")
    second = E5Atom(1, FEATURE_NAMES[1], "Identity")
    result = search_e5(
        train,
        train_target,
        validation,
        validation_truth,
        source_names=FEATURE_NAMES,
        settings=E5SearchSettings(max_distinct_valid_evaluations=2),
        candidate_proposals=(
            E5Structure((first,)),
            E5Structure((first, second)),
        ),
    )
    best_nrmse = min(
        candidate.validation_clean_nrmse for candidate in result.candidates
    )
    assert 0 < result.selected_validation_nrmse <= best_nrmse + 0.005
    assert len(result.selected_model.structure.canonical_atoms()) == 1
    assert result.selected_model.complexity()["ast_node_count"] == 6


def test_e5_source_mass_and_manifest_last_export_are_auditable(tmp_path) -> None:
    from mirage_kan.experiments.gate_a.data import FEATURE_NAMES
    from mirage_kan.experiments.gate_a.e5 import (
        E5Atom,
        E5ExecutableModel,
        E5SearchSettings,
        E5Structure,
        load_e5_export,
        save_e5_search,
        search_e5,
    )

    first = E5Atom(0, FEATURE_NAMES[0], "Identity")
    second = E5Atom(1, FEATURE_NAMES[1], "Identity")
    model = E5ExecutableModel(
        E5Structure((first, second)),
        np.array([2.0, -1.0]),
        300.0,
    )
    features = np.zeros((4, 6), dtype=np.float64)
    features[:, 0] = [-2.0, -1.0, 1.0, 2.0]
    features[:, 1] = [-1.0, 0.0, 2.0, 4.0]
    mass = model.source_mass(features)
    assert mass["status"] == "defined"
    assert sum(item["mass"] for item in mass["masses"]) == pytest.approx(1.0)
    shifted = features.copy()
    shifted[:, 0] += 1000.0
    shifted[:, 1] -= 250.0
    assert model.source_mass(shifted) == mass
    zero_model = E5ExecutableModel(
        model.structure, np.zeros(2, dtype=np.float64), 10.0
    )
    zero_mass = zero_model.source_mass(features)
    assert zero_mass["status"] == "zero_contribution_energy"
    assert all(item["mass"] == 0.0 for item in zero_mass["masses"])
    grouped_model = E5ExecutableModel(
        E5Structure(
            (
                first,
                E5Atom(0, FEATURE_NAMES[0], "Square"),
                second,
            )
        ),
        np.array([1.0, 0.5, -1.0]),
        0.0,
    )
    grouped = grouped_model.source_mass(features)
    source0 = features[:, 0] + 0.5 * np.square(features[:, 0])
    source1 = -features[:, 1]
    expected_energies = np.array(
        [np.var(source0, ddof=0), np.var(source1, ddof=0)]
    )
    assert grouped["masses"][0]["mass"] == pytest.approx(
        expected_energies[0] / expected_energies.sum()
    )

    target = model.evaluate(features)
    result = search_e5(
        features,
        target,
        shifted,
        model.evaluate(shifted),
        source_names=FEATURE_NAMES,
        settings=E5SearchSettings(max_distinct_valid_evaluations=1, seed=42),
        candidate_proposals=(model.structure,),
    )
    manifest_path = save_e5_search(
        result,
        tmp_path / "e5" / "literal_run",
        metadata={"smoke_scope": "train_validation_only", "test_evaluated": False},
    )
    manifest = json.loads(manifest_path.read_text())
    ledger_path = Path(manifest["paths"]["ledger"])
    model_path = Path(manifest["paths"]["selected_model"])
    assert ledger_path.is_file() and model_path.is_file()
    assert manifest_path.stat().st_mtime_ns >= ledger_path.stat().st_mtime_ns
    assert manifest_path.stat().st_mtime_ns >= model_path.stat().st_mtime_ns
    assert manifest["selected"]["source_mass"]["status"] == "defined"
    assert manifest["generation_mode"] == "caller_supplied_audit_proposals"
    restored = load_e5_export(manifest_path)
    np.testing.assert_array_equal(
        restored.evaluate(features), result.selected_model.evaluate(features)
    )
    fake_spec = tmp_path / "tampered_e5_spec.json"
    fake_spec.write_text("{}\n", encoding="utf-8")
    manifest["model_spec"]["resolved_path"] = str(fake_spec)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="model-spec hash mismatch"):
        load_e5_export(manifest_path)
    with pytest.raises(FileExistsError):
        save_e5_search(result, tmp_path / "e5" / "literal_run")


def test_e5_smoke_is_fresh_reduced_and_train_validation_only(tmp_path) -> None:
    import yaml

    from mirage_kan.experiments.gate_a.e5_smoke import run_e5_smoke

    sealed = yaml.safe_load(
        Path("configs/experiments/s1_gate_a_v0.yaml").read_text()
    )
    manifest_path = run_e5_smoke(
        sealed,
        run_id="e5_unit_smoke",
        seed=8675309,
        candidate_budget=2,
        artifact_root=tmp_path / "e5_smoke",
    )
    manifest = json.loads(manifest_path.read_text())
    expected_metadata = {
        "smoke_scope": "train_validation_only",
        "scientific_evidence": False,
        "test_evaluated": False,
        "seed": 8675309,
        "candidate_budget": 2,
    }
    for name, expected in expected_metadata.items():
        assert manifest["metadata"][name] == expected
    assert manifest["metadata"]["accessed_dataset_fields"] == [
        "train",
        "validation",
    ]
    assert manifest["accounting"]["distinct_valid_ast_evaluations"] == 2
    assert manifest["generation_mode"] == "deterministic_frozen_generator"
    assert "test" not in manifest["paths"]
    with pytest.raises(ValueError, match="frozen scientific seed"):
        run_e5_smoke(
            sealed,
            run_id="forbidden",
            seed=1729,
            candidate_budget=2,
            artifact_root=tmp_path / "e5_smoke",
        )
