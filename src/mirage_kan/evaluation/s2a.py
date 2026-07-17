"""Frozen S2a replay validation, decision rules, and human report."""

from __future__ import annotations

import hashlib
import math
from decimal import Decimal
from typing import Mapping

import pandas as pd


ARMS = (
    "alpha158_replay",
    "heterogeneous_selected",
    "random_typed",
    "label_permutation_selected",
)


def _finite_metric(payload: Mapping[str, object], key: str) -> float:
    value = float(payload[key])
    if not math.isfinite(value):
        raise ValueError(f"S2a metric is not finite: {key}")
    return value


def replay_anchor_checks(
    metrics: Mapping[str, object], protocol: Mapping[str, object]
) -> dict[str, object]:
    """Apply the three frozen absolute replay tolerances."""
    evaluation = protocol["evaluation"]
    anchor = evaluation["historical_anchor"]
    tolerance = evaluation["replay_tolerances"]
    specifications = {
        "information_ratio": (
            "information_ratio",
            "information_ratio_absolute",
        ),
        "max_drawdown": ("max_drawdown", "max_drawdown_absolute"),
        "rank_ic": ("Rank IC", "rank_ic_absolute"),
    }
    checks: dict[str, object] = {}
    for name, (metric_key, tolerance_key) in specifications.items():
        observed = _finite_metric(metrics, metric_key)
        observed_exact = Decimal(str(observed))
        expected_exact = Decimal(str(anchor[name]))
        maximum_exact = Decimal(str(tolerance[tolerance_key]))
        difference_exact = abs(observed_exact - expected_exact)
        checks[name] = {
            "observed": observed,
            "anchor": float(expected_exact),
            "absolute_difference": float(difference_exact),
            "tolerance": float(maximum_exact),
            "passed": difference_exact <= maximum_exact,
        }
    checks["passed"] = all(record["passed"] for record in checks.values())
    return checks


def _decimal_metric(payload: Mapping[str, object], key: str) -> Decimal:
    return Decimal(str(_finite_metric(payload, key)))


def decide_s2a(
    evaluations: Mapping[str, Mapping[str, object]],
    daily: Mapping[str, pd.DataFrame],
    protocol: Mapping[str, object],
    expected_calendar: pd.DatetimeIndex,
) -> dict[str, object]:
    """Compute every preregistered S2a signal and guardrail without discretion."""
    if set(evaluations) != set(ARMS) or set(daily) != set(ARMS):
        raise ValueError("S2a decision requires the fixed four-arm topology")
    required_years = {2022, 2023, 2024, 2025}
    calendar = pd.DatetimeIndex(pd.to_datetime(expected_calendar)).sort_values()
    if (
        calendar.empty
        or not calendar.is_unique
        or not calendar.is_monotonic_increasing
        or set(calendar.year) != required_years
    ):
        raise ValueError("S2a locked trading calendar is invalid")
    calendar_bytes = "\n".join(value.isoformat() for value in calendar).encode()
    calendar_evidence = {
        "date_count": len(calendar),
        "first_date": calendar[0].isoformat(),
        "last_date": calendar[-1].isoformat(),
        "sha256": hashlib.sha256(calendar_bytes).hexdigest(),
    }
    required_columns = {"daily_excess_return", "turnover", "realized_cost"}
    for arm, frame in daily.items():
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.empty:
            raise ValueError(f"S2a {arm} daily diagnostics need a nonempty DatetimeIndex")
        if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise ValueError(f"S2a {arm} daily index must be unique and monotonic")
        if not required_columns.issubset(frame.columns):
            raise ValueError(f"S2a {arm} daily diagnostics lack required columns")
        values = frame[list(required_columns)].to_numpy(dtype=float)
        if not math.isfinite(float(values.sum())):
            raise ValueError(f"S2a {arm} daily diagnostics contain non-finite values")
        if (frame[["turnover", "realized_cost"]] < 0).any().any():
            raise ValueError(f"S2a {arm} turnover or cost is negative")
        if not frame.index.equals(calendar):
            raise ValueError(f"S2a {arm} does not match the locked trading calendar")
    replay = replay_anchor_checks(evaluations["alpha158_replay"]["metrics"], protocol)
    headline = {
        arm: {
            "information_ratio": _finite_metric(
                evaluations[arm]["metrics"], "information_ratio"
            ),
            "max_drawdown": _finite_metric(
                evaluations[arm]["metrics"], "max_drawdown"
            ),
            "rank_ic": _finite_metric(evaluations[arm]["metrics"], "Rank IC"),
        }
        for arm in ARMS
    }
    if not replay["passed"]:
        return {
            "outcome": "s2a_inconclusive_infrastructure",
            "formal_promotion_allowed": False,
            "replay_anchor_checks": replay,
            "headline_metrics": headline,
            "criteria": {},
            "all_criteria_passed": False,
            "trading_calendar": calendar_evidence,
        }
    metrics = {arm: evaluations[arm]["metrics"] for arm in ARMS}
    selected = metrics["heterogeneous_selected"]
    replay_metrics = metrics["alpha158_replay"]
    random_metrics = metrics["random_typed"]
    permutation_metrics = metrics["label_permutation_selected"]
    thresholds = protocol["s2a_decision"]["advance_only_if_all_hold"]
    selected_daily = daily["heterogeneous_selected"]
    replay_daily = daily["alpha158_replay"]
    selected_turnover = Decimal(str(float(selected_daily["turnover"].mean())))
    replay_turnover = Decimal(str(float(replay_daily["turnover"].mean())))
    selected_cost = Decimal(str(float(selected_daily["realized_cost"].mean())))
    replay_cost = Decimal(str(float(replay_daily["realized_cost"].mean())))
    if replay_turnover == 0:
        raise ValueError("S2a replay mean turnover is zero; ratio is undefined")
    if replay_cost == 0:
        raise ValueError("S2a replay mean realized cost is zero; ratio is undefined")
    mean_turnover_ratio = selected_turnover / replay_turnover
    mean_cost_ratio = selected_cost / replay_cost
    selected_year = selected_daily["daily_excess_return"].groupby(
        pd.to_datetime(selected_daily.index).year
    ).sum()
    replay_year = replay_daily["daily_excess_return"].groupby(
        pd.to_datetime(replay_daily.index).year
    ).sum()
    calendar = pd.concat(
        {"selected": selected_year, "replay": replay_year}, axis=1, join="inner"
    ).dropna()
    years_not_worse = int((calendar["selected"] >= calendar["replay"]).sum())
    exact_values = {
        "selected_information_ratio_minimum": _decimal_metric(
            selected, "information_ratio"
        ),
        "selected_minus_replay_information_ratio_minimum": _decimal_metric(
            selected, "information_ratio"
        )
        - _decimal_metric(replay_metrics, "information_ratio"),
        "selected_minus_random_information_ratio_minimum": _decimal_metric(
            selected, "information_ratio"
        )
        - _decimal_metric(random_metrics, "information_ratio"),
        "selected_minus_permutation_information_ratio_minimum": _decimal_metric(
            selected, "information_ratio"
        )
        - _decimal_metric(permutation_metrics, "information_ratio"),
        "selected_minus_replay_max_drawdown_minimum": _decimal_metric(
            selected, "max_drawdown"
        )
        - _decimal_metric(replay_metrics, "max_drawdown"),
        "selected_minus_replay_rank_ic_minimum": _decimal_metric(
            selected, "Rank IC"
        )
        - _decimal_metric(replay_metrics, "Rank IC"),
        "selected_to_replay_mean_turnover_ratio_maximum": mean_turnover_ratio,
        "selected_to_replay_mean_cost_ratio_maximum": mean_cost_ratio,
        "selected_calendar_years_not_worse_than_replay_minimum": Decimal(
            years_not_worse
        ),
    }
    criteria: dict[str, object] = {}
    for name, exact_observed in exact_values.items():
        exact_threshold = Decimal(str(thresholds[name]))
        comparison = "at_most" if name.endswith("_maximum") else "at_least"
        passed = (
            exact_observed <= exact_threshold
            if comparison == "at_most"
            else exact_observed >= exact_threshold
        )
        criteria[name] = {
            "observed": float(exact_observed),
            "threshold": float(exact_threshold),
            "comparison": comparison,
            "passed": bool(passed),
        }
    passed = all(record["passed"] for record in criteria.values())
    return {
        "outcome": "advance_s2_formal" if passed else "s2a_screen_fail",
        "formal_promotion_allowed": False,
        "replay_anchor_checks": replay,
        "headline_metrics": headline,
        "criteria": criteria,
        "all_criteria_passed": passed,
        "calendar_active_return": {
            str(year): {
                "selected": float(row["selected"]),
                "replay": float(row["replay"]),
                "selected_not_worse": bool(row["selected"] >= row["replay"]),
            }
            for year, row in calendar.iterrows()
        },
        "trading_calendar": calendar_evidence,
    }


def chinese_report(decision: Mapping[str, object]) -> str:
    """Render the immutable machine decision as a readable Chinese report."""
    outcome = decision["outcome"]
    icon = {
        "advance_s2_formal": "✅",
        "s2a_screen_fail": "🛑",
        "s2a_inconclusive_infrastructure": "⚠️",
    }[outcome]
    labels = {
        "selected_information_ratio_minimum": "精选库净 IR",
        "selected_minus_replay_information_ratio_minimum": "精选库 − Alpha158 回放 IR",
        "selected_minus_random_information_ratio_minimum": "精选库 − 随机库 IR",
        "selected_minus_permutation_information_ratio_minimum": "精选库 − 标签置换库 IR",
        "selected_minus_replay_max_drawdown_minimum": "精选库 − 回放 MDD",
        "selected_minus_replay_rank_ic_minimum": "精选库 − 回放 RankIC",
        "selected_to_replay_mean_turnover_ratio_maximum": "精选库/回放平均换手率",
        "selected_to_replay_mean_cost_ratio_maximum": "精选库/回放平均实付成本",
        "selected_calendar_years_not_worse_than_replay_minimum": "主动收益不差于回放的年份数",
    }
    lines = [
        "# MIRAGE-KAN S2a 完整链路报告",
        "",
        f"## {icon} 结论：{outcome}",
        "",
        "本阶段是 2022–2025 开发集筛选，不是最终样本外确认。无论结果如何，"
        "均不允许直接发布正式因子库价值结论，图控制模块仍保持锁定。",
        "",
        "缩写说明：IR 表示信息比率；MDD 表示最大回撤；RankIC 表示每日横截面"
        "排序相关系数；S2a 表示 Plan C 的小规模完整链路开发筛选。",
        "",
        "## 🧭 Alpha158 回放锚点",
        "",
        "| 指标 | 当前回放 | 历史锚点 | 绝对差 | 容差 | 结果 |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for name, record in decision["replay_anchor_checks"].items():
        if name != "passed":
            lines.append(
                f"| {name} | {record['observed']:.6g} | {record['anchor']:.6g} | "
                f"{record['absolute_difference']:.6g} | {record['tolerance']:.6g} | "
                f"{'✅' if record['passed'] else '❌'} |"
            )
    lines.extend(
        [
            "",
            "## 📊 四臂核心指标",
            "",
            "| 实验臂 | IR | MDD | RankIC |",
            "|---|---:|---:|---:|",
        ]
    )
    for arm, metrics in decision["headline_metrics"].items():
        lines.append(
            f"| {arm} | {metrics['information_ratio']:.6g} | "
            f"{metrics['max_drawdown']:.6g} | {metrics['rank_ic']:.6g} |"
        )
    lines.extend(
        [
            "",
        "## 📏 冻结判据",
        "",
        "| 判据 | 观测值 | 阈值 | 结果 |",
        "|---|---:|---:|:---:|",
        ]
    )
    criteria = decision.get("criteria", {})
    if criteria:
        for name, record in criteria.items():
            lines.append(
                f"| {labels[name]} | {record['observed']:.6g} | "
                f"{record['threshold']:.6g} | "
                f"{'✅' if record['passed'] else '❌'} |"
            )
    else:
        lines.append("| Alpha158 回放身份/锚点 | — | 冻结容差 | ❌ |")
    lines.extend(
        [
            "",
            "## 📅 分年度主动收益",
            "",
            "| 年份 | 精选库 | 回放 | 不差于回放 |",
            "|---:|---:|---:|:---:|",
        ]
    )
    for year, record in decision.get("calendar_active_return", {}).items():
        lines.append(
            f"| {year} | {record['selected']:.6g} | {record['replay']:.6g} | "
            f"{'✅' if record['selected_not_worse'] else '❌'} |"
        )
    lines.extend(
        [
            "",
            "## 🔒 证据边界",
            "",
            "- formal_promotion_allowed = false：本报告不能产生正式晋级或论文结论。",
            "- 所有四臂共享一次不可替换的开发测试打开记录。",
            "- 只有全部九项冻结判据通过，才可进入更大规模 S2 formal 实验。",
            "",
        ]
    )
    return "\n".join(lines)
