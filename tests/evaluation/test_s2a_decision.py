from __future__ import annotations

import pandas as pd
import pytest

from mirage_kan.evaluation.s2a import (
    ARMS,
    chinese_report,
    decide_s2a,
    replay_anchor_checks,
)


def _protocol() -> dict[str, object]:
    return {
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
        },
        "s2a_decision": {
            "advance_only_if_all_hold": {
                "selected_information_ratio_minimum": 0.32,
                "selected_minus_replay_information_ratio_minimum": 0.10,
                "selected_minus_random_information_ratio_minimum": 0.05,
                "selected_minus_permutation_information_ratio_minimum": 0.05,
                "selected_minus_replay_max_drawdown_minimum": -0.03,
                "selected_minus_replay_rank_ic_minimum": -0.003,
                "selected_to_replay_mean_turnover_ratio_maximum": 1.20,
                "selected_to_replay_mean_cost_ratio_maximum": 1.20,
                "selected_calendar_years_not_worse_than_replay_minimum": 3,
            }
        },
    }


def _evidence() -> tuple[dict, dict]:
    metrics = {
        "alpha158_replay": (0.22, -0.1376, 0.03311),
        "heterogeneous_selected": (0.34, -0.15, 0.032),
        "random_typed": (0.25, -0.16, 0.02),
        "label_permutation_selected": (0.25, -0.17, 0.01),
    }
    evaluations = {
        arm: {
            "metrics": {
                "information_ratio": values[0],
                "max_drawdown": values[1],
                "Rank IC": values[2],
            }
        }
        for arm, values in metrics.items()
    }
    index = pd.to_datetime(["2022-06-01", "2023-06-01", "2024-06-01", "2025-06-01"])
    daily = {}
    for arm in ARMS:
        excess = [0.01, 0.01, 0.01, 0.01]
        if arm == "heterogeneous_selected":
            excess = [0.02, 0.02, 0.02, 0.0]
        daily[arm] = pd.DataFrame(
            {"daily_excess_return": excess, "turnover": 0.1, "realized_cost": 0.001},
            index=index,
        )
    return evaluations, daily


def test_decision_applies_all_nine_frozen_criteria_and_human_report() -> None:
    evaluations, daily = _evidence()
    decision = decide_s2a(
        evaluations, daily, _protocol(), daily["alpha158_replay"].index
    )

    assert decision["outcome"] == "advance_s2_formal"
    assert len(decision["criteria"]) == 9
    assert decision["formal_promotion_allowed"] is False
    report = chinese_report(decision)
    assert "四臂核心指标" in report
    assert "Alpha158 回放锚点" in report
    assert "分年度主动收益" in report
    assert "精选库净 IR" in report


def test_decision_rejects_missing_daily_dates_even_when_all_years_present() -> None:
    evaluations, daily = _evidence()
    daily["random_typed"] = daily["random_typed"].iloc[:-1]
    with pytest.raises(ValueError, match="locked trading calendar"):
        decide_s2a(
            evaluations, daily, _protocol(), daily["alpha158_replay"].index
        )


def test_replay_miss_uses_frozen_infrastructure_outcome() -> None:
    evaluations, daily = _evidence()
    evaluations["alpha158_replay"]["metrics"]["information_ratio"] = -1.0
    decision = decide_s2a(
        evaluations, daily, _protocol(), daily["alpha158_replay"].index
    )
    assert decision["outcome"] == "s2a_inconclusive_infrastructure"
    assert decision["criteria"] == {}


def _boundary_evidence() -> tuple[dict, dict]:
    evaluations, daily = _evidence()
    evaluations["heterogeneous_selected"]["metrics"].update(
        {"information_ratio": 0.32, "max_drawdown": -0.1676, "Rank IC": 0.03011}
    )
    evaluations["random_typed"]["metrics"]["information_ratio"] = 0.27
    evaluations["label_permutation_selected"]["metrics"]["information_ratio"] = 0.27
    daily["heterogeneous_selected"]["turnover"] = 0.12
    daily["heterogeneous_selected"]["realized_cost"] = 0.0012
    return evaluations, daily


def test_all_nine_exact_threshold_boundaries_are_inclusive() -> None:
    evaluations, daily = _boundary_evidence()
    decision = decide_s2a(
        evaluations, daily, _protocol(), daily["alpha158_replay"].index
    )
    assert decision["outcome"] == "advance_s2_formal"
    assert all(record["passed"] for record in decision["criteria"].values())


def test_frozen_thresholds_have_no_unregistered_numeric_tolerance() -> None:
    evaluations, daily = _boundary_evidence()
    evaluations["heterogeneous_selected"]["metrics"]["information_ratio"] = (
        0.32 - 5e-13
    )

    decision = decide_s2a(
        evaluations, daily, _protocol(), daily["alpha158_replay"].index
    )

    criterion = decision["criteria"]["selected_information_ratio_minimum"]
    assert criterion["passed"] is False
    assert decision["outcome"] == "s2a_screen_fail"


def test_replay_tolerances_are_exact_and_inclusive() -> None:
    protocol = _protocol()
    boundary = {
        "information_ratio": 0.25,
        "max_drawdown": -0.1576,
        "Rank IC": 0.03611,
    }
    assert replay_anchor_checks(boundary, protocol)["passed"] is True

    boundary["information_ratio"] = 0.25 + 5e-13
    checks = replay_anchor_checks(boundary, protocol)
    assert checks["information_ratio"]["passed"] is False
    assert checks["passed"] is False


def test_zero_replay_denominator_is_an_infrastructure_error() -> None:
    evaluations, daily = _evidence()
    daily["alpha158_replay"]["turnover"] = 0.0

    with pytest.raises(ValueError, match="replay mean turnover is zero"):
        decide_s2a(
            evaluations, daily, _protocol(), daily["alpha158_replay"].index
        )


def test_decision_rejects_a_shared_but_incomplete_trading_calendar() -> None:
    evaluations, daily = _evidence()
    expected = daily["alpha158_replay"].index.append(
        pd.DatetimeIndex([pd.Timestamp("2025-07-01")])
    ).sort_values()

    with pytest.raises(ValueError, match="locked trading calendar"):
        decide_s2a(evaluations, daily, _protocol(), expected)


@pytest.mark.parametrize(
    "criterion",
    [
        "selected_information_ratio_minimum",
        "selected_minus_replay_information_ratio_minimum",
        "selected_minus_random_information_ratio_minimum",
        "selected_minus_permutation_information_ratio_minimum",
        "selected_minus_replay_max_drawdown_minimum",
        "selected_minus_replay_rank_ic_minimum",
        "selected_to_replay_mean_turnover_ratio_maximum",
        "selected_to_replay_mean_cost_ratio_maximum",
        "selected_calendar_years_not_worse_than_replay_minimum",
    ],
)
def test_each_frozen_criterion_fails_just_beyond_its_boundary(criterion: str) -> None:
    evaluations, daily = _boundary_evidence()
    epsilon = 1e-8
    if criterion == "selected_information_ratio_minimum":
        evaluations["heterogeneous_selected"]["metrics"]["information_ratio"] = 0.32 - epsilon
    elif criterion == "selected_minus_replay_information_ratio_minimum":
        evaluations["alpha158_replay"]["metrics"]["information_ratio"] = 0.22 + epsilon
    elif criterion == "selected_minus_random_information_ratio_minimum":
        evaluations["random_typed"]["metrics"]["information_ratio"] = 0.27 + epsilon
    elif criterion == "selected_minus_permutation_information_ratio_minimum":
        evaluations["label_permutation_selected"]["metrics"]["information_ratio"] = 0.27 + epsilon
    elif criterion == "selected_minus_replay_max_drawdown_minimum":
        evaluations["heterogeneous_selected"]["metrics"]["max_drawdown"] = -0.1676 - epsilon
    elif criterion == "selected_minus_replay_rank_ic_minimum":
        evaluations["heterogeneous_selected"]["metrics"]["Rank IC"] = 0.03011 - epsilon
    elif criterion == "selected_to_replay_mean_turnover_ratio_maximum":
        daily["heterogeneous_selected"]["turnover"] = 0.12 + epsilon
    elif criterion == "selected_to_replay_mean_cost_ratio_maximum":
        daily["heterogeneous_selected"]["realized_cost"] = 0.0012 + epsilon
    else:
        daily["heterogeneous_selected"]["daily_excess_return"] = [0.02, 0.02, 0.0, 0.0]

    decision = decide_s2a(
        evaluations, daily, _protocol(), daily["alpha158_replay"].index
    )
    assert decision["outcome"] == "s2a_screen_fail"
    assert decision["criteria"][criterion]["passed"] is False
