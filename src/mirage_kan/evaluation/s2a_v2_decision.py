"""Pure S2a v2 five-arm bootstrap and machine-decision logic."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from mirage_kan.evaluation.s2a import replay_anchor_checks

ARMS = (
    "alpha158_replay",
    "kan_e3_selected",
    "typed_gp_sr_control",
    "matched_blackbox_control",
    "kan_e3_permutation_control",
)
REQUIRED_DAILY_COLUMNS = (
    "daily_excess_return",
    "turnover",
    "realized_cost",
)


def _information_ratio(values: np.ndarray) -> float:
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("information-ratio input must be finite and one-dimensional")
    standard_deviation = float(np.std(values, ddof=1))
    if not math.isfinite(standard_deviation) or standard_deviation <= 0.0:
        raise ValueError("bootstrap resample has zero or non-finite variance")
    value = math.sqrt(252.0) * float(np.mean(values)) / standard_deviation
    if not math.isfinite(value):
        raise ValueError("bootstrap information ratio is non-finite")
    return value


def paired_block_bootstrap_delta_ir(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
    *,
    seed: int,
    block_length: int,
    replicates: int,
    confidence_level: float = 0.95,
) -> dict[str, object]:
    """Return a deterministic paired non-wrapping moving-block delta-IR sample."""
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if (
        left_values.ndim != 1
        or right_values.ndim != 1
        or left_values.shape != right_values.shape
        or not np.isfinite(left_values).all()
        or not np.isfinite(right_values).all()
    ):
        raise ValueError("paired bootstrap inputs must be same-shape finite vectors")
    if type(block_length) is not int or block_length < 1 or block_length > len(left_values):
        raise ValueError("block length must be within the paired calendar")
    if type(replicates) is not int or replicates < 1:
        raise ValueError("bootstrap replicates must be a positive integer")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("bootstrap confidence level must be between zero and one")

    observed_delta = _information_ratio(left_values) - _information_ratio(right_values)
    random = np.random.Generator(np.random.PCG64(int(seed)))
    starts_count = len(left_values) - block_length + 1
    blocks_needed = math.ceil(len(left_values) / block_length)
    offsets = np.arange(block_length, dtype=np.int64)
    deltas: list[float] = []
    for _ in range(replicates):
        starts = random.integers(0, starts_count, size=blocks_needed)
        indices = (starts[:, None] + offsets[None, :]).reshape(-1)[: len(left_values)]
        delta = _information_ratio(left_values[indices]) - _information_ratio(
            right_values[indices]
        )
        if not math.isfinite(delta):
            raise ValueError("bootstrap delta information ratio is non-finite")
        deltas.append(delta)
    lower_quantile = 1.0 - confidence_level
    return {
        "seed": int(seed),
        "block_length": block_length,
        "replicate_count": replicates,
        "confidence_level": float(confidence_level),
        "paired": True,
        "annualization": "sqrt_252",
        "standard_deviation_ddof": 1,
        "observed_delta_ir": observed_delta,
        "lower_confidence_bound": float(np.quantile(deltas, lower_quantile)),
        "replicate_delta_ir": deltas,
    }


def _finite_metric(metrics: Mapping[str, object], name: str) -> float:
    value = float(metrics[name])
    if not math.isfinite(value):
        raise ValueError(f"S2a v2 metric is non-finite: {name}")
    return value


def _criterion(
    observed: float | int,
    threshold: float | int,
    comparison: str,
    passed: bool,
) -> dict[str, object]:
    return {
        "observed": observed,
        "threshold": threshold,
        "comparison": comparison,
        "passed": bool(passed),
    }


def _graph_unlock_allowed(protocol: Mapping[str, object]) -> bool:
    claim_boundary = protocol.get("claim_boundary")
    if not isinstance(claim_boundary, Mapping):
        raise ValueError("graph unlock claim boundary must be a mapping")
    value = claim_boundary.get("graph_unlock_allowed")
    if type(value) is not bool:
        raise ValueError("graph unlock permission must be an explicit boolean")
    return value


def _validate_daily(
    daily: Mapping[str, pd.DataFrame], expected_calendar: pd.DatetimeIndex
) -> pd.DatetimeIndex:
    calendar = pd.DatetimeIndex(pd.to_datetime(expected_calendar))
    if (
        calendar.empty
        or not calendar.is_unique
        or not calendar.is_monotonic_increasing
        or set(calendar.year) != {2022, 2023, 2024, 2025}
    ):
        raise ValueError("S2a v2 expected calendar is invalid")
    for arm, frame in daily.items():
        if not isinstance(frame, pd.DataFrame) or not frame.index.equals(calendar):
            raise ValueError(f"S2a v2 {arm} daily diagnostics do not match calendar")
        if not set(REQUIRED_DAILY_COLUMNS).issubset(frame.columns):
            raise ValueError(f"S2a v2 {arm} daily diagnostics lack required columns")
        values = frame.loc[:, REQUIRED_DAILY_COLUMNS].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"S2a v2 {arm} daily diagnostics are non-finite")
        if (frame[["turnover", "realized_cost"]] < 0.0).any().any():
            raise ValueError(f"S2a v2 {arm} turnover or cost is negative")
    return calendar


def _bootstrap(
    daily: Mapping[str, pd.DataFrame],
    left: str,
    right: str,
    settings: Mapping[str, object],
) -> dict[str, object]:
    return paired_block_bootstrap_delta_ir(
        daily[left]["daily_excess_return"].to_numpy(dtype=float),
        daily[right]["daily_excess_return"].to_numpy(dtype=float),
        seed=int(settings["seed"]),
        block_length=int(settings["block_length_trading_days"]),
        replicates=int(settings["replicates"]),
        confidence_level=float(settings["confidence_level"]),
    )


def _human_review_status(
    evidence: Mapping[str, object],
    protocol: Mapping[str, object],
    library_size: int,
) -> str:
    review = evidence.get("human_blind_review")
    if not isinstance(review, Mapping):
        raise ValueError("human blind-review evidence must be a mapping")
    status = review.get("status")
    if status not in {"pending", "complete", "failed"}:
        raise ValueError("unknown human blind-review status")
    if status != "complete":
        return str(status)

    settings = protocol["interpretability"]["human_blind_review"]
    reviewers_minimum = int(settings["reviewers_minimum"])
    accuracy_minimum = float(settings["response_direction_accuracy_minimum"])
    reviews = review.get("reviews")
    if not isinstance(reviews, list) or len(reviews) < reviewers_minimum:
        raise ValueError("complete review lacks the required human reviewers")
    package_hash = evidence.get("blind_review_package_sha256")
    if (
        not isinstance(package_hash, str)
        or len(package_hash) != 64
        or any(character not in "0123456789abcdef" for character in package_hash)
    ):
        raise ValueError("blind-review package identity is invalid")
    reviewer_ids: set[str] = set()
    for record in reviews:
        if not isinstance(record, Mapping):
            raise ValueError("human review record must be a mapping")
        reviewer_id = record.get("reviewer_id")
        if not isinstance(reviewer_id, str) or not reviewer_id:
            raise ValueError("human review lacks reviewer identity")
        if reviewer_id in reviewer_ids:
            raise ValueError("human reviewers must be unique")
        reviewer_ids.add(reviewer_id)
        if (
            record.get("reviewer_type") != "human_quantitative_reviewer"
            or record.get("human_attestation") is not True
        ):
            raise ValueError("complete review lacks explicit human attestation")
        restatements = record.get("mechanism_restatements")
        if (
            not isinstance(restatements, Mapping)
            or len(restatements) != library_size
            or not all(
                isinstance(value, str) and bool(value.strip())
                for value in restatements.values()
            )
        ):
            raise ValueError("human review mechanism restatements are incomplete")
        correct = record.get("response_direction_correct")
        total = record.get("response_direction_total")
        if (
            type(correct) is not int
            or type(total) is not int
            or total < 1
            or not 0 <= correct <= total
            or correct / total < accuracy_minimum
        ):
            raise ValueError("human review response-direction accuracy is below gate")
        if record.get("reviewed_blind_package_sha256") != package_hash:
            raise ValueError("human review used a different blind package")
    agreement = review.get("inter_reviewer_agreement")
    if (
        not isinstance(agreement, (int, float))
        or isinstance(agreement, bool)
        or not math.isfinite(float(agreement))
        or not 0.0 <= float(agreement) <= 1.0
    ):
        raise ValueError("inter-reviewer agreement was not validly reported")
    return "complete"


def decide_s2a_v2(
    evaluations: Mapping[str, Mapping[str, object]],
    daily: Mapping[str, pd.DataFrame],
    evidence: Mapping[str, object],
    protocol: Mapping[str, object],
    expected_calendar: pd.DatetimeIndex,
) -> dict[str, object]:
    """Apply every frozen v2 infrastructure, performance, and integrity gate."""
    if set(evaluations) != set(ARMS) or set(daily) != set(ARMS):
        raise ValueError("S2a v2 decision requires the exact five-arm topology")
    graph_unlock_allowed = _graph_unlock_allowed(protocol)
    calendar = _validate_daily(daily, expected_calendar)
    metrics = {
        arm: evaluations[arm]["metrics"]
        for arm in ARMS
    }
    if any(not isinstance(value, Mapping) for value in metrics.values()):
        raise ValueError("S2a v2 evaluation metrics must be mappings")
    replay = replay_anchor_checks(metrics["alpha158_replay"], protocol)
    headline = {
        arm: {
            "information_ratio": _finite_metric(metrics[arm], "information_ratio"),
            "max_drawdown": _finite_metric(metrics[arm], "max_drawdown"),
            "rank_ic": _finite_metric(metrics[arm], "Rank IC"),
        }
        for arm in ARMS
    }
    calendar_evidence = {
        "date_count": len(calendar),
        "first_date": calendar[0].isoformat(),
        "last_date": calendar[-1].isoformat(),
    }
    if not replay["passed"]:
        return {
            "outcome": "s2a_inconclusive_infrastructure",
            "all_machine_criteria_passed": False,
            "formal_promotion_allowed": False,
            "graph_unlock_allowed": graph_unlock_allowed,
            "replay_anchor_checks": replay,
            "headline_metrics": headline,
            "criteria": {},
            "bootstrap": {},
            "trading_calendar": calendar_evidence,
        }

    decision = protocol["s2a_decision"]
    performance = decision["performance_noninferiority"]
    integrity = decision["integrity"]
    bootstrap_settings = protocol["evaluation"]["bootstrap"]
    kan_alpha = _bootstrap(
        daily, "kan_e3_selected", "alpha158_replay", bootstrap_settings
    )
    kan_blackbox = _bootstrap(
        daily, "kan_e3_selected", "matched_blackbox_control", bootstrap_settings
    )
    gp_alpha = _bootstrap(
        daily, "typed_gp_sr_control", "alpha158_replay", bootstrap_settings
    )

    kan_daily = daily["kan_e3_selected"]
    alpha_daily = daily["alpha158_replay"]
    alpha_turnover = float(alpha_daily["turnover"].mean())
    alpha_cost = float(alpha_daily["realized_cost"].mean())
    if alpha_turnover <= 0.0 or alpha_cost <= 0.0:
        raise ValueError("Alpha158 turnover and cost means must be positive")
    turnover_ratio = float(kan_daily["turnover"].mean()) / alpha_turnover
    cost_ratio = float(kan_daily["realized_cost"].mean()) / alpha_cost
    kan_year = kan_daily["daily_excess_return"].groupby(calendar.year).sum()
    alpha_year = alpha_daily["daily_excess_return"].groupby(calendar.year).sum()
    years_not_worse = int((kan_year >= alpha_year).sum())

    library_size = int(evidence["production_library_size"])
    control_sizes = (
        int(evidence["gp_control_library_size"]),
        int(evidence["blackbox_control_output_count"]),
        int(evidence["permutation_control_library_size"]),
    )
    blackbox_margin = -0.1 * abs(
        headline["matched_blackbox_control"]["information_ratio"]
    )
    criteria: dict[str, dict[str, object]] = {}
    alpha_lcb_threshold = float(
        performance["kan_minus_alpha158_delta_ir_lcb_minimum"]
    )
    criteria["kan_alpha_delta_ir_lcb"] = _criterion(
        kan_alpha["lower_confidence_bound"],
        alpha_lcb_threshold,
        "at_least",
        float(kan_alpha["lower_confidence_bound"]) >= alpha_lcb_threshold,
    )
    criteria["kan_blackbox_delta_ir_lcb"] = _criterion(
        kan_blackbox["lower_confidence_bound"],
        blackbox_margin,
        "at_least",
        float(kan_blackbox["lower_confidence_bound"]) >= blackbox_margin,
    )
    rank_ic_delta = (
        headline["kan_e3_selected"]["rank_ic"]
        - headline["alpha158_replay"]["rank_ic"]
    )
    rank_ic_threshold = float(performance["kan_minus_alpha158_rank_ic_minimum"])
    criteria["kan_alpha_rank_ic_delta"] = _criterion(
        rank_ic_delta,
        rank_ic_threshold,
        "at_least",
        rank_ic_delta >= rank_ic_threshold,
    )
    drawdown_delta = (
        headline["kan_e3_selected"]["max_drawdown"]
        - headline["alpha158_replay"]["max_drawdown"]
    )
    drawdown_threshold = float(
        performance["kan_minus_alpha158_max_drawdown_minimum"]
    )
    criteria["kan_alpha_max_drawdown_delta"] = _criterion(
        drawdown_delta,
        drawdown_threshold,
        "at_least",
        drawdown_delta >= drawdown_threshold,
    )
    turnover_threshold = float(
        performance["kan_to_alpha158_mean_turnover_ratio_maximum"]
    )
    criteria["kan_alpha_turnover_ratio"] = _criterion(
        turnover_ratio,
        turnover_threshold,
        "at_most",
        turnover_ratio <= turnover_threshold,
    )
    cost_threshold = float(performance["kan_to_alpha158_mean_cost_ratio_maximum"])
    criteria["kan_alpha_cost_ratio"] = _criterion(
        cost_ratio, cost_threshold, "at_most", cost_ratio <= cost_threshold
    )
    years_threshold = int(
        performance["calendar_years_not_worse_than_alpha158_minimum"]
    )
    criteria["calendar_years_not_worse"] = _criterion(
        years_not_worse,
        years_threshold,
        "at_least",
        years_not_worse >= years_threshold,
    )

    minimum_size = int(integrity["production_library_size_minimum"])
    maximum_size = int(protocol.get("admission", {}).get("library_cap", 16))
    criteria["production_library_size"] = _criterion(
        library_size,
        minimum_size,
        f"between_{minimum_size}_and_{maximum_size}",
        minimum_size <= library_size <= maximum_size,
    )
    profile_count = int(evidence["production_profile_count"])
    minimum_profiles = int(integrity["production_library_profiles_minimum"])
    criteria["production_profile_count"] = _criterion(
        profile_count,
        minimum_profiles,
        "at_least",
        profile_count >= minimum_profiles,
    )
    fraction_checks = {
        "production_strict_fraction": float(
            integrity["production_factors_strict_fraction"]
        ),
        "production_mechanism_card_fraction": float(
            integrity["production_factors_with_mechanism_card_fraction"]
        ),
        "production_kan_mined_fraction": 1.0,
        "production_lineage_fraction": 1.0,
        "production_independent_replay_fraction": 1.0,
    }
    for name, threshold in fraction_checks.items():
        observed = float(evidence[name])
        if not math.isfinite(observed):
            raise ValueError(f"S2a v2 evidence is non-finite: {name}")
        criteria[name] = _criterion(
            observed, threshold, "at_least", observed >= threshold
        )
    maximum_spline = float(evidence["production_max_spline_ratio"])
    criteria["production_max_spline_ratio"] = _criterion(
        maximum_spline, 0.0, "equal", maximum_spline == 0.0
    )
    criteria["control_size_match"] = _criterion(
        list(control_sizes),
        library_size,
        "all_equal_production_size",
        all(size == library_size for size in control_sizes),
    )
    false_positives = int(evidence["permutation_false_positive_count"])
    false_positive_maximum = int(
        integrity["permutation_false_positive_count_maximum"]
    )
    criteria["permutation_false_positive_count"] = _criterion(
        false_positives,
        false_positive_maximum,
        "at_most",
        false_positives <= false_positive_maximum,
    )

    tolerance = decision["method_falsification"]["numerical_tolerance"]
    ir_tolerance = float(tolerance["delta_ir_lcb"])
    rank_tolerance = float(tolerance["effective_rank"])
    kan_ir_lcb = float(kan_alpha["lower_confidence_bound"])
    gp_ir_lcb = float(gp_alpha["lower_confidence_bound"])
    kan_admitted = int(evidence["kan_unique_admitted_count"])
    gp_admitted = int(evidence["gp_unique_admitted_count"])
    kan_rank = float(evidence["kan_selected_library_effective_rank"])
    gp_rank = float(evidence["gp_selected_library_effective_rank"])
    if not all(math.isfinite(value) for value in (kan_rank, gp_rank)):
        raise ValueError("S2a v2 effective-rank evidence is non-finite")
    greater_or_equal = (
        gp_ir_lcb >= kan_ir_lcb - ir_tolerance,
        gp_admitted >= kan_admitted,
        gp_rank >= kan_rank - rank_tolerance,
    )
    strictly_greater = (
        gp_ir_lcb > kan_ir_lcb + ir_tolerance,
        gp_admitted > kan_admitted,
        gp_rank > kan_rank + rank_tolerance,
    )
    gp_dominates = all(greater_or_equal) and any(strictly_greater)
    criteria["gp_does_not_pareto_dominate"] = {
        "observed": {
            "kan": [kan_ir_lcb, kan_admitted / 256.0, kan_rank],
            "gp": [gp_ir_lcb, gp_admitted / 256.0, gp_rank],
        },
        "threshold": "gp_not_ge_all_with_one_strict",
        "comparison": "not_pareto_dominated",
        "passed": not gp_dominates,
    }

    all_machine = all(item["passed"] for item in criteria.values())
    review_status = _human_review_status(evidence, protocol, library_size)
    if not all_machine or review_status == "failed":
        outcome = "s2a_screen_fail"
    elif review_status == "complete":
        outcome = "advance_s2_formal"
    else:
        outcome = "advance_s2_formal_pending_human_blind_review"
    return {
        "outcome": outcome,
        "all_machine_criteria_passed": all_machine,
        "human_blind_review_status": review_status,
        "formal_promotion_allowed": False,
        "graph_unlock_allowed": graph_unlock_allowed,
        "replay_anchor_checks": replay,
        "headline_metrics": headline,
        "criteria": criteria,
        "bootstrap": {
            "kan_minus_alpha158": kan_alpha,
            "kan_minus_blackbox": kan_blackbox,
            "gp_minus_alpha158": gp_alpha,
        },
        "calendar_active_return": {
            str(year): {
                "kan": float(kan_year.loc[year]),
                "alpha158": float(alpha_year.loc[year]),
                "kan_not_worse": bool(kan_year.loc[year] >= alpha_year.loc[year]),
            }
            for year in sorted(set(calendar.year))
        },
        "trading_calendar": calendar_evidence,
    }


__all__ = ["ARMS", "decide_s2a_v2", "paired_block_bootstrap_delta_ir"]
