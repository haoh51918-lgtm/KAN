from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest


ARMS = (
    "alpha158_replay",
    "kan_e3_selected",
    "typed_gp_sr_control",
    "matched_blackbox_control",
    "kan_e3_permutation_control",
)


def _calendar() -> pd.DatetimeIndex:
    values = []
    for year in (2022, 2023, 2024, 2025):
        values.extend(pd.bdate_range(f"{year}-01-03", periods=30))
    return pd.DatetimeIndex(values)


def _protocol() -> dict[str, object]:
    return {
        "claim_boundary": {"graph_unlock_allowed": True},
        "evaluation": {
            "historical_anchor": {
                "information_ratio": 0.22,
                "max_drawdown": -0.1376,
                "rank_ic": 0.03311,
            },
            "replay_tolerances": {
                "information_ratio_absolute": 0.03,
                "max_drawdown_absolute": 0.02,
                "rank_ic_absolute": 0.003,
            },
            "bootstrap": {
                "seed": 67867967,
                "block_length_trading_days": 20,
                "replicates": 200,
                "confidence_level": 0.95,
            },
        },
        "s2a_decision": {
            "performance_noninferiority": {
                "kan_minus_alpha158_delta_ir_lcb_minimum": -0.022,
                "kan_minus_blackbox_delta_ir_lcb_minimum_formula": (
                    "-0.1_times_absolute_blackbox_ir"
                ),
                "kan_minus_alpha158_rank_ic_minimum": -0.003,
                "kan_minus_alpha158_max_drawdown_minimum": -0.03,
                "kan_to_alpha158_mean_turnover_ratio_maximum": 1.2,
                "kan_to_alpha158_mean_cost_ratio_maximum": 1.2,
                "calendar_years_not_worse_than_alpha158_minimum": 3,
            },
            "integrity": {
                "production_library_size_minimum": 8,
                "production_library_profiles_minimum": 3,
                "production_factors_strict_fraction": 1.0,
                "production_factors_with_mechanism_card_fraction": 1.0,
                "permutation_false_positive_count_maximum": 0,
                "alpha158_replay_must_match_anchor": True,
            },
            "method_falsification": {
                "numerical_tolerance": {
                    "delta_ir_lcb": 1e-12,
                    "effective_rank": 1e-12,
                }
            },
        },
        "interpretability": {
            "human_blind_review": {
                "reviewers_minimum": 2,
                "mechanism_restatement_required": True,
                "response_direction_accuracy_minimum": 0.8,
                "inter_reviewer_agreement_reported": True,
            }
        },
    }


def _inputs() -> tuple[
    dict[str, dict[str, object]],
    dict[str, pd.DataFrame],
    dict[str, object],
    pd.DatetimeIndex,
]:
    calendar = _calendar()
    wave = np.sin(np.arange(len(calendar), dtype=float) / 4.0) * 0.001
    means = {
        "alpha158_replay": 0.00020,
        "kan_e3_selected": 0.00045,
        "typed_gp_sr_control": 0.00025,
        "matched_blackbox_control": 0.00032,
        "kan_e3_permutation_control": 0.00005,
    }
    daily = {
        arm: pd.DataFrame(
            {
                "daily_excess_return": wave + mean,
                "turnover": 0.10,
                "realized_cost": 0.001,
            },
            index=calendar,
        )
        for arm, mean in means.items()
    }
    metrics = {
        "alpha158_replay": {
            "information_ratio": 0.22,
            "max_drawdown": -0.1376,
            "Rank IC": 0.03311,
        },
        "kan_e3_selected": {
            "information_ratio": 0.40,
            "max_drawdown": -0.13,
            "Rank IC": 0.034,
        },
        "typed_gp_sr_control": {
            "information_ratio": 0.25,
            "max_drawdown": -0.14,
            "Rank IC": 0.031,
        },
        "matched_blackbox_control": {
            "information_ratio": 0.32,
            "max_drawdown": -0.13,
            "Rank IC": 0.034,
        },
        "kan_e3_permutation_control": {
            "information_ratio": 0.05,
            "max_drawdown": -0.15,
            "Rank IC": 0.001,
        },
    }
    evaluations = {arm: {"metrics": metrics[arm]} for arm in ARMS}
    evidence = {
        "production_library_size": 12,
        "production_profile_count": 4,
        "production_strict_fraction": 1.0,
        "production_mechanism_card_fraction": 1.0,
        "production_kan_mined_fraction": 1.0,
        "production_lineage_fraction": 1.0,
        "production_independent_replay_fraction": 1.0,
        "production_max_spline_ratio": 0.0,
        "gp_control_library_size": 12,
        "blackbox_control_output_count": 12,
        "permutation_control_library_size": 12,
        "permutation_false_positive_count": 0,
        "kan_unique_admitted_count": 20,
        "gp_unique_admitted_count": 15,
        "kan_selected_library_effective_rank": 8.0,
        "gp_selected_library_effective_rank": 6.0,
        "blind_review_package_sha256": "c" * 64,
        "human_blind_review": {"status": "pending", "reviews": []},
    }
    return evaluations, daily, evidence, calendar


def test_paired_bootstrap_is_reproducible_and_uses_paired_indices() -> None:
    from mirage_kan.evaluation.s2a_v2_decision import paired_block_bootstrap_delta_ir

    left = np.array([0.01, -0.01, 0.02, 0.00, 0.03, -0.02], dtype=float)
    right = left * 0.5
    first = paired_block_bootstrap_delta_ir(
        left, right, seed=17, block_length=2, replicates=25, confidence_level=0.95
    )
    second = paired_block_bootstrap_delta_ir(
        left, right, seed=17, block_length=2, replicates=25, confidence_level=0.95
    )

    assert first == second
    assert first["replicate_count"] == 25
    assert first["block_length"] == 2
    assert first["paired"] is True
    assert first["replicate_delta_ir"] == second["replicate_delta_ir"]
    assert first["observed_delta_ir"] == pytest.approx(0.0)


def test_machine_pass_with_pending_human_review_has_distinct_outcome() -> None:
    from mirage_kan.evaluation.s2a_v2_decision import decide_s2a_v2

    evaluations, daily, evidence, calendar = _inputs()
    result = decide_s2a_v2(evaluations, daily, evidence, _protocol(), calendar)

    assert result["outcome"] == "advance_s2_formal_pending_human_blind_review"
    assert result["all_machine_criteria_passed"] is True
    assert result["formal_promotion_allowed"] is False
    assert result["graph_unlock_allowed"] is True
    assert result["criteria"]["kan_alpha_delta_ir_lcb"]["passed"] is True
    assert result["criteria"]["kan_blackbox_delta_ir_lcb"]["passed"] is True
    assert result["criteria"]["gp_does_not_pareto_dominate"]["passed"] is True
    assert set(result["headline_metrics"]) == set(ARMS)


def test_completed_human_review_changes_only_success_label() -> None:
    from mirage_kan.evaluation.s2a_v2_decision import decide_s2a_v2

    evaluations, daily, evidence, calendar = _inputs()
    evidence["human_blind_review"] = {
        "status": "complete",
        "inter_reviewer_agreement": 0.9,
        "reviews": [
            {
                "reviewer_id": reviewer_id,
                "reviewer_type": "human_quantitative_reviewer",
                "human_attestation": True,
                "mechanism_restatements": {
                    f"factor_{index:02d}": "Restated mechanism"
                    for index in range(12)
                },
                "response_direction_correct": 9,
                "response_direction_total": 10,
                "reviewed_blind_package_sha256": "c" * 64,
            }
            for reviewer_id in ("reviewer_a", "reviewer_b")
        ],
    }
    result = decide_s2a_v2(evaluations, daily, evidence, _protocol(), calendar)

    assert result["outcome"] == "advance_s2_formal"
    assert result["all_machine_criteria_passed"] is True


def test_complete_human_review_cannot_be_asserted_by_a_status_string_or_weak_review() -> None:
    from mirage_kan.evaluation.s2a_v2_decision import decide_s2a_v2

    evaluations, daily, evidence, calendar = _inputs()
    evidence["human_blind_review"] = "complete"
    with pytest.raises(ValueError, match="review evidence"):
        decide_s2a_v2(evaluations, daily, evidence, _protocol(), calendar)

    evaluations, daily, evidence, calendar = _inputs()
    evidence["human_blind_review"] = {
        "status": "complete",
        "inter_reviewer_agreement": 0.9,
        "reviews": [
            {
                "reviewer_id": reviewer_id,
                "reviewer_type": "human_quantitative_reviewer",
                "human_attestation": True,
                "mechanism_restatements": {
                    f"factor_{index:02d}": "Restated mechanism"
                    for index in range(12)
                },
                "response_direction_correct": 7,
                "response_direction_total": 10,
                "reviewed_blind_package_sha256": "c" * 64,
            }
            for reviewer_id in ("reviewer_a", "reviewer_b")
        ],
    }
    with pytest.raises(ValueError, match="accuracy"):
        decide_s2a_v2(evaluations, daily, evidence, _protocol(), calendar)


def test_replay_miss_is_infrastructure_inconclusive_before_scientific_decision() -> None:
    from mirage_kan.evaluation.s2a_v2_decision import decide_s2a_v2

    evaluations, daily, evidence, calendar = _inputs()
    evaluations["alpha158_replay"]["metrics"]["information_ratio"] = 0.50
    result = decide_s2a_v2(evaluations, daily, evidence, _protocol(), calendar)

    assert result["outcome"] == "s2a_inconclusive_infrastructure"
    assert result["all_machine_criteria_passed"] is False
    assert result["graph_unlock_allowed"] is True
    assert result["criteria"] == {}


def test_graph_unlock_requires_an_explicit_boolean_claim_boundary() -> None:
    from mirage_kan.evaluation.s2a_v2_decision import decide_s2a_v2

    evaluations, daily, evidence, calendar = _inputs()
    protocol = _protocol()
    protocol["claim_boundary"]["graph_unlock_allowed"] = "true"

    with pytest.raises(ValueError, match="graph unlock"):
        decide_s2a_v2(evaluations, daily, evidence, protocol, calendar)


def test_gp_pareto_dominance_and_false_positive_each_fail_closed() -> None:
    from mirage_kan.evaluation.s2a_v2_decision import decide_s2a_v2

    evaluations, daily, evidence, calendar = _inputs()
    daily["typed_gp_sr_control"]["daily_excess_return"] = (
        daily["kan_e3_selected"]["daily_excess_return"] + 0.001
    )
    evidence["gp_unique_admitted_count"] = 21
    evidence["gp_selected_library_effective_rank"] = 9.0
    dominated = decide_s2a_v2(evaluations, daily, evidence, _protocol(), calendar)
    assert dominated["outcome"] == "s2a_screen_fail"
    assert dominated["criteria"]["gp_does_not_pareto_dominate"]["passed"] is False

    evaluations, daily, evidence, calendar = _inputs()
    evidence["permutation_false_positive_count"] = 1
    false_positive = decide_s2a_v2(
        evaluations, daily, evidence, _protocol(), calendar
    )
    assert false_positive["outcome"] == "s2a_screen_fail"
    assert false_positive["criteria"]["permutation_false_positive_count"][
        "passed"
    ] is False


def test_control_sizes_and_production_provenance_are_integrity_gates() -> None:
    from mirage_kan.evaluation.s2a_v2_decision import decide_s2a_v2

    evaluations, daily, evidence, calendar = _inputs()
    evidence["production_kan_mined_fraction"] = 11 / 12
    evidence["blackbox_control_output_count"] = 11
    result = decide_s2a_v2(evaluations, daily, evidence, _protocol(), calendar)

    assert result["outcome"] == "s2a_screen_fail"
    assert result["criteria"]["production_kan_mined_fraction"]["passed"] is False
    assert result["criteria"]["control_size_match"]["passed"] is False


def test_wrong_arms_calendar_or_degenerate_bootstrap_fail() -> None:
    from mirage_kan.evaluation.s2a_v2_decision import (
        decide_s2a_v2,
        paired_block_bootstrap_delta_ir,
    )

    evaluations, daily, evidence, calendar = _inputs()
    missing = deepcopy(evaluations)
    missing.pop("matched_blackbox_control")
    with pytest.raises(ValueError, match="five-arm"):
        decide_s2a_v2(missing, daily, evidence, _protocol(), calendar)

    bad_daily = deepcopy(daily)
    bad_daily["kan_e3_selected"] = bad_daily["kan_e3_selected"].iloc[:-1]
    with pytest.raises(ValueError, match="calendar"):
        decide_s2a_v2(evaluations, bad_daily, evidence, _protocol(), calendar)

    with pytest.raises(ValueError, match="variance"):
        paired_block_bootstrap_delta_ir(
            np.ones(30), np.ones(30), seed=1, block_length=20, replicates=5
        )
