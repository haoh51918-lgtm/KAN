from __future__ import annotations

import pytest
import torch


def _validation_atom_panel(manifest):
    import numpy as np
    import pandas as pd

    from mirage_kan.mining.e3_runner import AtomPanel

    dates = pd.date_range("2021-01-04", periods=2, freq="B")
    instruments = pd.Index(["A", "B", "C"], name="instrument")
    index = pd.MultiIndex.from_product(
        (dates, instruments), names=("datetime", "instrument")
    )
    values = np.zeros((2, 3, len(manifest)), dtype=np.float64)
    values[..., 4] = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    values[..., 33] = np.array([[6.0, 4.0, 1.0], [3.0, 2.0, 0.5]])
    support = np.ones_like(values, dtype=bool)
    membership = np.ones((2, 3), dtype=bool)
    return AtomPanel(
        profile="price_volume",
        atom_manifest=tuple(manifest),
        dates=dates,
        instruments=instruments,
        index=index,
        values=values,
        support=support,
        joint_support=support.all(axis=-1),
        membership=membership,
    )


def _receipt_and_manifest():
    from mirage_kan.mining.e3 import (
        build_profile_atom_bank,
        harden_checkpoint,
    )

    manifest = build_profile_atom_bank("price_volume")
    logits = torch.full((2, len(manifest)), -3.0, dtype=torch.float64)
    logits[0, 4] = 6.0
    logits[1, 33] = 5.0
    return harden_checkpoint(logits, manifest, 0.1), manifest


def _card():
    from mirage_kan.artifacts.mechanism import build_mechanism_card

    receipt, manifest = _receipt_and_manifest()
    return build_mechanism_card(
        factor_id="factor_07",
        profile="price_volume",
        receipt=receipt,
        atom_manifest=manifest,
        fidelity={"pearson": 0.997, "nrmse": 0.021, "max_absolute_error": 0.04},
        variable_interventions={"Close": 0.014, "Volume": -0.006},
        lag_interventions={"window_5": 0.009, "window_20": -0.004},
        local_counterfactual_response={
            "positive_atom_plus_one_sigma": 0.12,
            "negative_atom_plus_one_sigma": -0.08,
        },
        lineage={
            "miner": "kan_e3",
            "global_seed": 733001,
            "checkpoint_sha256": "a" * 64,
            "hardening_receipt_sha256": "b" * 64,
        },
    )


def test_mechanism_card_has_every_frozen_field_and_exact_ast_facts() -> None:
    card = _card()

    assert set(card) == {
        "identity_and_canonical_ast",
        "raw_variables_windows_and_complexity",
        "edge_gate_and_shape_summary",
        "soft_hard_fidelity",
        "variable_and_lag_interventions",
        "local_counterfactual_response",
        "one_sentence_mechanism",
        "applicability_and_failure_conditions",
        "complete_kan_lineage_and_hardening_receipt",
    }
    identity = card["identity_and_canonical_ast"]
    assert identity["factor_id"] == "factor_07"
    assert identity["canonical_hash"] == identity["ast_identity"]
    assert identity["canonical_ast"].startswith("{")
    facts = card["raw_variables_windows_and_complexity"]
    assert set(facts["raw_variables"]).issubset(
        {"Open", "High", "Low", "Close", "Volume"}
    )
    assert facts["windows"]
    assert facts["ast_nodes"] >= 3
    assert facts["ast_depth"] >= 2
    assert facts["required_lookback"] >= max(facts["windows"])
    assert "positive edge" in card["one_sentence_mechanism"]
    assert "negative edge" in card["one_sentence_mechanism"]


def test_card_retains_checkpoint_gate_and_lineage_evidence_without_claiming_human_review() -> (
    None
):
    card = _card()

    edge = card["edge_gate_and_shape_summary"]
    assert edge["positive"]["probability"] > edge["positive"]["runner_up_probability"]
    assert edge["negative"]["margin"] > 0.0
    shape = edge["shape_plot_data"]
    for edge_name in ("positive_edge", "negative_edge"):
        edge_shape = shape[edge_name]
        assert len(edge_shape["primitive_input_ratio"]) == 101
        assert len(edge_shape["primitive_output"]) == 101
        assert len(edge_shape["edge_contribution"]) == 101
        assert edge_shape["monotonic_direction"] in {"increasing", "decreasing"}
        assert edge_shape["saturation_intervals"] == []
    lineage = card["complete_kan_lineage_and_hardening_receipt"]
    assert lineage["miner"] == "kan_e3"
    assert lineage["checkpoint_logits_sha256"]
    assert lineage["atom_manifest_sha256"]
    assert "human" not in str(card).lower()


def test_validation_interventions_are_exact_selected_edge_ablations() -> None:
    import numpy as np

    from mirage_kan.artifacts.mechanism import compute_mechanism_interventions

    receipt, manifest = _receipt_and_manifest()
    panel = _validation_atom_panel(manifest)
    variable, lag, local = compute_mechanism_interventions(receipt, panel)

    baseline = panel.values[..., 4] - panel.values[..., 33]
    scale = float(np.std(baseline, ddof=0))
    assert variable["Low.mean_absolute_delta_over_baseline_std"] == pytest.approx(
        float(np.mean(np.abs(panel.values[..., 4]))) / scale
    )
    assert variable["High.mean_absolute_delta_over_baseline_std"] == pytest.approx(
        float(np.mean(np.abs(panel.values[..., 33]))) / scale
    )
    assert lag["short_2_5.mean_absolute_delta_over_baseline_std"] == pytest.approx(
        float(np.mean(np.abs(baseline))) / scale
    )
    assert lag["short_2_5.baseline_intervened_pearson"] == 0.0
    assert lag["medium_10_20.mean_absolute_delta_over_baseline_std"] == 0.0
    assert lag["medium_10_20.baseline_intervened_pearson"] == 1.0
    assert local["positive_plus_one_sigma_response_sign"] == 1.0
    assert local["positive_minus_one_sigma_response_sign"] == -1.0
    assert local["negative_plus_one_sigma_response_sign"] == -1.0
    assert local["negative_minus_one_sigma_response_sign"] == 1.0


def test_interventions_reject_nonvalidation_rows_and_degenerate_evidence() -> None:
    from dataclasses import replace

    import pandas as pd

    from mirage_kan.artifacts.mechanism import compute_mechanism_interventions

    receipt, manifest = _receipt_and_manifest()
    panel = _validation_atom_panel(manifest)
    panel = replace(panel, dates=pd.date_range("2022-01-03", periods=2, freq="B"))
    with pytest.raises(ValueError, match="validation period"):
        compute_mechanism_interventions(receipt, panel)

    panel = _validation_atom_panel(manifest)
    panel.values[..., 4] = panel.values[..., 33]
    with pytest.raises(ValueError, match="baseline population standard deviation"):
        compute_mechanism_interventions(receipt, panel)


def test_blind_package_is_deterministic_and_hides_method_lineage_and_returns() -> None:
    from mirage_kan.artifacts.mechanism import build_blind_review_package

    card = _card()
    package = build_blind_review_package(
        {"factor_07": card, "factor_02": card},
        hides=("method_name", "pnl", "return_metrics"),
        reviewers_minimum=2,
        response_direction_accuracy_minimum=0.8,
    )

    assert package["review_status"] == "pending_human_review"
    assert package["reviewers_minimum"] == 2
    assert package["response_direction_accuracy_minimum"] == 0.8
    assert list(package["items"]) == ["B001", "B002"]
    serialized = str(package).lower()
    assert "factor_07" not in serialized
    assert "factor_02" not in serialized
    assert "kan_e3" not in serialized
    assert "checkpoint" not in serialized
    assert "probability" not in serialized
    assert "soft_hard" not in serialized
    assert "positive_atom_plus_one_sigma" not in serialized
    assert "negative_atom_plus_one_sigma" not in serialized
    assert "pnl" not in serialized
    assert "return_metrics" not in serialized
    assert len(package["items"]["B001"]["response_direction_questions"]) == 2
    assert package["items"]["B001"]["identity_and_canonical_ast"]["canonical_ast"]


def test_nonfinite_or_mismatched_evidence_fails_closed() -> None:
    from mirage_kan.artifacts.mechanism import build_mechanism_card

    receipt, manifest = _receipt_and_manifest()
    with pytest.raises(ValueError, match="finite"):
        build_mechanism_card(
            factor_id="factor_07",
            profile="price_volume",
            receipt=receipt,
            atom_manifest=manifest,
            fidelity={"pearson": float("nan"), "nrmse": 0.1},
            variable_interventions={"Close": 0.1},
            lag_interventions={"window_5": 0.1},
            local_counterfactual_response={"up": 0.1},
            lineage={"miner": "kan_e3"},
        )

    with pytest.raises(ValueError, match="manifest"):
        build_mechanism_card(
            factor_id="factor_07",
            profile="short_price",
            receipt=receipt,
            atom_manifest=manifest,
            fidelity={"pearson": 1.0, "nrmse": 0.0},
            variable_interventions={"Close": 0.1},
            lag_interventions={"window_5": 0.1},
            local_counterfactual_response={"up": 0.1},
            lineage={"miner": "kan_e3"},
        )
