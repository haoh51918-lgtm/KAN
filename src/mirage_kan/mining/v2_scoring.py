"""Shared hard-AST scoring, admission, selection, and diversity for S2a v2."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from mirage_kan.data import PitPanel
from mirage_kan.dsl import AstNode, DslType, ProgramError, evaluate
from mirage_kan.mining.e3 import PROFILE_SPECS, build_profile_atom_bank

METHODS = frozenset({"kan_e3", "typed_gp_sr"})


def _ast_node_count(ast: AstNode) -> int:
    return 1 + sum(_ast_node_count(child) for child in ast.children)


def _ast_depth(ast: AstNode) -> int:
    return 1 + max((_ast_depth(child) for child in ast.children), default=0)


def _series_sha256(series: pd.Series) -> str:
    digest = hashlib.sha256()
    digest.update(str(series.name).encode())
    digest.update(pd.util.hash_pandas_object(series, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def _date_mask(index: pd.MultiIndex, bounds: tuple[str, str]) -> np.ndarray:
    dates = pd.to_datetime(index.get_level_values("datetime"))
    return (dates >= pd.Timestamp(bounds[0])) & (dates <= pd.Timestamp(bounds[1]))


def _mean_daily_spearman(left: pd.Series, right: pd.Series) -> float | None:
    frame = pd.concat({"left": left, "right": right}, axis=1).dropna()
    correlations: list[float] = []
    for _, daily in frame.groupby(level="datetime", sort=False):
        if (
            len(daily) < 2
            or daily["left"].nunique() < 2
            or daily["right"].nunique() < 2
        ):
            continue
        correlation = daily["left"].corr(daily["right"], method="spearman")
        if np.isfinite(correlation):
            correlations.append(float(correlation))
    return float(np.mean(correlations)) if correlations else None


def _fidelity(
    soft: pd.Series, hard: pd.Series, mask: pd.Series
) -> tuple[float, float] | None:
    aligned_soft = soft.reindex(hard.index)
    finite = pd.Series(
        np.isfinite(aligned_soft.to_numpy(dtype=float)), index=hard.index
    ) & pd.Series(np.isfinite(hard.to_numpy(dtype=float)), index=hard.index)
    usable = mask & finite
    soft_values = aligned_soft.loc[usable].to_numpy(dtype=float)
    hard_values = hard.loc[usable].to_numpy(dtype=float)
    if len(soft_values) < 2:
        return None
    centered_soft = soft_values - soft_values.mean()
    centered_hard = hard_values - hard_values.mean()
    scale = math.sqrt(float(np.sum(centered_soft**2)) * float(np.sum(centered_hard**2)))
    if scale == 0.0:
        pearson = 1.0 if np.array_equal(soft_values, hard_values) else 0.0
    else:
        pearson = float(np.sum(centered_soft * centered_hard) / scale)
    denominator = float(np.std(soft_values, ddof=0))
    rmse = float(np.sqrt(np.mean((soft_values - hard_values) ** 2)))
    nrmse = (
        0.0
        if denominator == 0.0 and rmse == 0.0
        else (float("inf") if denominator == 0.0 else rmse / denominator)
    )
    return pearson, nrmse


@dataclass(frozen=True)
class HardAstCandidate:
    """One independently replayable formula plus method-specific lineage evidence."""

    candidate_id: str
    profile: str
    attempt_index: int
    ast: AstNode | None
    method: str
    lineage_complete: bool
    soft_values: pd.Series | None = field(default=None, repr=False, compare=False)
    gate_probability_margin: float | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty")
        if self.profile not in PROFILE_SPECS:
            raise ValueError(f"unknown E3 profile: {self.profile!r}")
        if self.method not in METHODS:
            raise ValueError(f"unknown hard-AST method: {self.method!r}")
        if isinstance(self.attempt_index, bool) or self.attempt_index < 0:
            raise ValueError("attempt_index must be a nonnegative integer")


@dataclass(frozen=True)
class HardAstScore:
    """Common structural, support, efficacy, fidelity, and lineage gate record."""

    candidate_id: str
    profile: str
    attempt_index: int
    method: str
    ast: AstNode | None
    canonical_hash: str | None
    ast_depth: int | None
    ast_nodes: int | None
    output_type: str | None
    causal: bool
    unique: bool
    support_rows: int
    eligible_rows: int
    coverage: float
    train_rank_ic: float | None
    validation_rank_ic: float | None
    sign_agreement: bool
    minimum_score: float | None
    fidelity_pearson: float | None
    fidelity_nrmse: float | None
    gate_probability_margin: float | None
    fidelity_gate_met: bool
    lineage_gate_met: bool
    production_eligible: bool
    null_eligible: bool
    production_disposition: str
    null_disposition: str
    validation_values: pd.Series = field(repr=False, compare=False)

    def to_record(self) -> dict[str, object]:
        values = {
            key: value
            for key, value in self.__dict__.items()
            if key not in {"ast", "validation_values"}
        }
        values["ast"] = self.ast.to_dict() if self.ast is not None else None
        return values


@dataclass(frozen=True)
class HardAstScoringRun:
    """One common scoring pass bound to exact labels and split boundaries."""

    candidates: tuple[HardAstScore, ...]
    label_mode: str
    label_sha256: str
    scoring_run_id: str
    train: tuple[str, str]
    validation: tuple[str, str]

    @property
    def production_admitted_count(self) -> int:
        return sum(candidate.production_eligible for candidate in self.candidates)


@dataclass(frozen=True)
class HardAstSelection:
    """Selected order and an explicit terminal disposition for every candidate."""

    selected: tuple[HardAstScore, ...]
    dispositions: dict[str, str]
    minimum_size_met: bool
    exact_size_met: bool
    profile_quota_met: bool
    target_size: int | None

    @property
    def complete(self) -> bool:
        return self.minimum_size_met and self.exact_size_met and self.profile_quota_met

    def require_complete(self) -> None:
        if not self.complete:
            raise RuntimeError(
                "hard-AST selection is incomplete: "
                f"selected={len(self.selected)}, target={self.target_size}, "
                f"minimum_size_met={self.minimum_size_met}, "
                f"profile_quota_met={self.profile_quota_met}"
            )


def _validate_thresholds(
    *,
    minimum_coverage: float,
    minimum_absolute_train_rank_ic: float,
    minimum_absolute_validation_rank_ic: float,
    minimum_soft_hard_pearson: float,
    maximum_soft_hard_nrmse: float,
    minimum_gate_probability_margin: float,
) -> None:
    unit_interval = {
        "minimum_coverage": minimum_coverage,
        "minimum_absolute_train_rank_ic": minimum_absolute_train_rank_ic,
        "minimum_absolute_validation_rank_ic": minimum_absolute_validation_rank_ic,
        "minimum_soft_hard_pearson": minimum_soft_hard_pearson,
        "minimum_gate_probability_margin": minimum_gate_probability_margin,
    }
    for name, value in unit_interval.items():
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    if not math.isfinite(maximum_soft_hard_nrmse) or maximum_soft_hard_nrmse < 0:
        raise ValueError("maximum_soft_hard_nrmse must be finite and nonnegative")


def _strict_ast_contract(
    candidate: HardAstCandidate,
) -> tuple[bool, str | None, int | None, int | None, str | None, bool]:
    ast = candidate.ast
    if ast is None:
        return False, None, None, None, None, False
    try:
        contract = ast.validate()
    except (ProgramError, ValueError):
        return False, None, None, None, None, False
    bank_hashes = {
        atom.canonical_hash for atom in build_profile_atom_bank(candidate.profile)
    }
    strict = (
        ast.op == "Sub"
        and len(ast.children) == 2
        and ast.children[0].identity in bank_hashes
        and ast.children[1].identity in bank_hashes
        and ast.children[0].identity != ast.children[1].identity
        and contract.output_type is DslType.DIMENSIONLESS_TS
        and contract.causal
        and _ast_depth(ast) <= 6
        and _ast_node_count(ast) <= 20
    )
    return (
        strict,
        ast.identity,
        _ast_depth(ast),
        _ast_node_count(ast),
        contract.output_type.value,
        contract.causal,
    )


def _resolve_canonical_duplicates(
    scores: Sequence[HardAstScore],
) -> list[HardAstScore]:
    groups: dict[str, list[int]] = {}
    for index, score in enumerate(scores):
        if score.unique and score.canonical_hash is not None:
            groups.setdefault(score.canonical_hash, []).append(index)
    resolved = list(scores)
    for indices in groups.values():
        if len(indices) < 2:
            continue
        winner = min(
            indices,
            key=lambda index: (
                -int(scores[index].production_eligible),
                -int(scores[index].null_eligible),
                -int(scores[index].lineage_gate_met),
                -int(scores[index].fidelity_gate_met),
                -float(scores[index].gate_probability_margin or -math.inf),
                scores[index].attempt_index,
                scores[index].candidate_id,
            ),
        )
        for index in indices:
            if index == winner:
                continue
            resolved[index] = replace(
                scores[index],
                unique=False,
                production_eligible=False,
                null_eligible=False,
                production_disposition="duplicate_canonical_identity",
                null_disposition="duplicate_canonical_identity",
            )
    return resolved


def score_hard_ast_candidates(
    candidates: Iterable[HardAstCandidate],
    panel: PitPanel,
    labels: pd.Series,
    *,
    train: tuple[str, str],
    validation: tuple[str, str],
    minimum_coverage: float,
    minimum_absolute_train_rank_ic: float,
    minimum_absolute_validation_rank_ic: float,
    minimum_soft_hard_pearson: float,
    maximum_soft_hard_nrmse: float,
    minimum_gate_probability_margin: float,
    label_mode: str,
) -> HardAstScoringRun:
    """Replay every AST once and apply shared production and null gates."""
    _validate_thresholds(
        minimum_coverage=minimum_coverage,
        minimum_absolute_train_rank_ic=minimum_absolute_train_rank_ic,
        minimum_absolute_validation_rank_ic=minimum_absolute_validation_rank_ic,
        minimum_soft_hard_pearson=minimum_soft_hard_pearson,
        maximum_soft_hard_nrmse=maximum_soft_hard_nrmse,
        minimum_gate_probability_margin=minimum_gate_probability_margin,
    )
    train_start, train_end = map(pd.Timestamp, train)
    validation_start, validation_end = map(pd.Timestamp, validation)
    if (
        train_start > train_end
        or validation_start > validation_end
        or train_end >= validation_start
    ):
        raise ValueError("train and validation boundaries must be ordered and disjoint")
    dates = pd.to_datetime(panel.raw.index.get_level_values("datetime"))
    if dates.max() > validation_end:
        raise ValueError("hard-AST scoring panel crosses the validation boundary")
    if labels.name != "fwd":
        raise ValueError("hard-AST scoring requires the exact fwd label")
    if not isinstance(labels.index, pd.MultiIndex):
        raise ValueError("labels must use the panel MultiIndex")
    candidate_rows = tuple(candidates)
    candidate_ids = [candidate.candidate_id for candidate in candidate_rows]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate_id values must be unique within one scoring run")

    labels = labels.reindex(panel.raw.index)
    label_hash = _series_sha256(labels)
    train_mask = pd.Series(_date_mask(panel.raw.index, train), index=panel.raw.index)
    validation_mask = pd.Series(
        _date_mask(panel.raw.index, validation), index=panel.raw.index
    )
    development_mask = train_mask | validation_mask
    finite_label = pd.Series(
        np.isfinite(labels.to_numpy(dtype=float)), index=panel.raw.index
    )
    denominator = panel.membership & finite_label & development_mask
    eligible_rows = int(denominator.sum())
    scores: list[HardAstScore] = []
    for candidate in candidate_rows:
        empty = pd.Series(np.nan, index=panel.raw.index[validation_mask], dtype=float)
        (
            strict,
            canonical_hash,
            depth,
            nodes,
            output_type,
            causal,
        ) = _strict_ast_contract(candidate)
        unique = bool(strict)
        if not strict or candidate.ast is None:
            scores.append(
                HardAstScore(
                    candidate.candidate_id,
                    candidate.profile,
                    candidate.attempt_index,
                    candidate.method,
                    candidate.ast,
                    canonical_hash,
                    depth,
                    nodes,
                    output_type,
                    causal,
                    unique,
                    0,
                    eligible_rows,
                    0.0,
                    None,
                    None,
                    False,
                    None,
                    None,
                    None,
                    candidate.gate_probability_margin,
                    False,
                    bool(candidate.lineage_complete),
                    False,
                    False,
                    "invalid_strict_ast",
                    "invalid_strict_ast",
                    empty,
                )
            )
            continue

        result = evaluate(candidate.ast, panel)
        factor_effective = result.support & panel.membership
        label_effective = factor_effective & finite_label
        support_rows = int((label_effective & development_mask).sum())
        coverage = support_rows / eligible_rows if eligible_rows else 0.0
        hard_values = result.values.where(label_effective)
        train_ic = _mean_daily_spearman(
            hard_values.where(train_mask), labels.where(train_mask)
        )
        validation_ic = _mean_daily_spearman(
            hard_values.where(validation_mask), labels.where(validation_mask)
        )
        sign_agreement = bool(
            train_ic is not None
            and validation_ic is not None
            and train_ic != 0.0
            and validation_ic != 0.0
            and np.sign(train_ic) == np.sign(validation_ic)
        )
        minimum_score = (
            min(abs(train_ic), abs(validation_ic))
            if train_ic is not None and validation_ic is not None
            else None
        )

        if candidate.method == "typed_gp_sr":
            fidelity_pearson, fidelity_nrmse = 1.0, 0.0
            fidelity_gate_met = True
        elif candidate.soft_values is None:
            fidelity_pearson, fidelity_nrmse = None, None
            fidelity_gate_met = False
        else:
            metrics = _fidelity(
                candidate.soft_values,
                result.values,
                result.support & panel.membership & development_mask,
            )
            if metrics is None:
                fidelity_pearson, fidelity_nrmse = None, None
                fidelity_gate_met = False
            else:
                fidelity_pearson, fidelity_nrmse = metrics
                fidelity_gate_met = bool(
                    fidelity_pearson >= minimum_soft_hard_pearson
                    and fidelity_nrmse <= maximum_soft_hard_nrmse
                )
        lineage_gate_met = bool(candidate.lineage_complete)
        margin_gate_met = bool(
            candidate.method == "typed_gp_sr"
            or (
                candidate.gate_probability_margin is not None
                and math.isfinite(candidate.gate_probability_margin)
                and candidate.gate_probability_margin >= minimum_gate_probability_margin
            )
        )

        production_structural_disposition = None
        if not lineage_gate_met:
            production_structural_disposition = "ineligible_lineage"
        elif coverage < minimum_coverage:
            production_structural_disposition = "ineligible_coverage"
        elif not fidelity_gate_met:
            production_structural_disposition = "ineligible_soft_hard_fidelity"
        elif not margin_gate_met:
            production_structural_disposition = "ineligible_gate_margin"

        if coverage < minimum_coverage:
            null_disposition = "ineligible_coverage"
        elif train_ic is None or validation_ic is None:
            null_disposition = "ineligible_null_score_undefined"
        else:
            null_disposition = "null_eligible"

        if production_structural_disposition is not None:
            production_disposition = production_structural_disposition
        else:
            if train_ic is None or abs(train_ic) < minimum_absolute_train_rank_ic:
                production_disposition = "ineligible_train_rank_ic"
            elif (
                validation_ic is None
                or abs(validation_ic) < minimum_absolute_validation_rank_ic
            ):
                production_disposition = "ineligible_validation_rank_ic"
            elif not sign_agreement:
                production_disposition = "ineligible_sign_disagreement"
            else:
                production_disposition = "production_eligible"
        scores.append(
            HardAstScore(
                candidate_id=candidate.candidate_id,
                profile=candidate.profile,
                attempt_index=candidate.attempt_index,
                method=candidate.method,
                ast=candidate.ast,
                canonical_hash=canonical_hash,
                ast_depth=depth,
                ast_nodes=nodes,
                output_type=output_type,
                causal=causal,
                unique=unique,
                support_rows=support_rows,
                eligible_rows=eligible_rows,
                coverage=coverage,
                train_rank_ic=train_ic,
                validation_rank_ic=validation_ic,
                sign_agreement=sign_agreement,
                minimum_score=minimum_score,
                fidelity_pearson=fidelity_pearson,
                fidelity_nrmse=fidelity_nrmse,
                gate_probability_margin=candidate.gate_probability_margin,
                fidelity_gate_met=fidelity_gate_met,
                lineage_gate_met=lineage_gate_met,
                production_eligible=production_disposition == "production_eligible",
                null_eligible=null_disposition == "null_eligible",
                production_disposition=production_disposition,
                null_disposition=null_disposition,
                validation_values=result.values.where(factor_effective).loc[
                    validation_mask
                ],
            )
        )

    scores = _resolve_canonical_duplicates(scores)
    payload = {
        "label_mode": label_mode,
        "label_sha256": label_hash,
        "train": train,
        "validation": validation,
        "candidates": [candidate.to_record() for candidate in scores],
    }
    scoring_run_id = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return HardAstScoringRun(
        tuple(scores), label_mode, label_hash, scoring_run_id, train, validation
    )


def _diversity_rejection(
    candidate: HardAstScore,
    selected: Sequence[HardAstScore],
    threshold: float,
) -> str | None:
    for existing in selected:
        correlation = _mean_daily_spearman(
            candidate.validation_values, existing.validation_values
        )
        if correlation is None:
            return "rejected_diversity_undefined"
        if abs(correlation) >= threshold - 1e-12:
            return "rejected_diversity"
    return None


def _select(
    candidates: Iterable[HardAstScore],
    *,
    eligibility: str,
    disposition_field: str,
    library_cap: int,
    minimum_library_size: int,
    minimum_profiles: int,
    maximum_absolute_validation_spearman: float,
    target_size: int | None,
) -> HardAstSelection:
    if not 1 <= minimum_library_size <= library_cap:
        raise ValueError("minimum library size must be in [1, library_cap]")
    if not 1 <= minimum_profiles <= len(PROFILE_SPECS):
        raise ValueError("minimum_profiles is outside the frozen profile count")
    if not 0.0 < maximum_absolute_validation_spearman <= 1.0:
        raise ValueError("diversity threshold must be in (0, 1]")
    if target_size is not None and target_size != library_cap:
        raise ValueError("size-matched selection target must equal its exact cap")
    rows = tuple(candidates)
    ids = [candidate.candidate_id for candidate in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("selection candidate IDs must be unique")
    dispositions = {
        candidate.candidate_id: str(getattr(candidate, disposition_field))
        for candidate in rows
    }
    ordered = sorted(
        (candidate for candidate in rows if bool(getattr(candidate, eligibility))),
        key=lambda candidate: (
            -float(candidate.minimum_score),
            int(candidate.ast_nodes),
            str(candidate.canonical_hash),
        ),
    )
    selected: list[HardAstScore] = []
    decided: set[str] = set()
    profiles: set[str] = set()

    for candidate in ordered:
        if len(profiles) >= minimum_profiles:
            break
        if candidate.profile in profiles:
            continue
        rejection = _diversity_rejection(
            candidate, selected, maximum_absolute_validation_spearman
        )
        if rejection is not None:
            dispositions[candidate.candidate_id] = rejection
            decided.add(candidate.candidate_id)
            continue
        selected.append(candidate)
        profiles.add(candidate.profile)
        dispositions[candidate.candidate_id] = "selected"
        decided.add(candidate.candidate_id)

    for candidate in ordered:
        if candidate.candidate_id in decided:
            continue
        if len(selected) >= library_cap:
            dispositions[candidate.candidate_id] = "eligible_not_selected_cap"
            decided.add(candidate.candidate_id)
            continue
        rejection = _diversity_rejection(
            candidate, selected, maximum_absolute_validation_spearman
        )
        if rejection is not None:
            dispositions[candidate.candidate_id] = rejection
        else:
            selected.append(candidate)
            profiles.add(candidate.profile)
            dispositions[candidate.candidate_id] = "selected"
        decided.add(candidate.candidate_id)

    exact_size_met = target_size is None or len(selected) == target_size
    return HardAstSelection(
        selected=tuple(selected),
        dispositions=dispositions,
        minimum_size_met=len(selected) >= minimum_library_size,
        exact_size_met=exact_size_met,
        profile_quota_met=len(profiles) >= minimum_profiles,
        target_size=target_size,
    )


def select_production_candidates(
    candidates: Iterable[HardAstScore],
    *,
    library_cap: int,
    minimum_library_size: int,
    minimum_profiles: int,
    maximum_absolute_validation_spearman: float,
) -> HardAstSelection:
    """Select a production or GP method-control library under real efficacy gates."""
    if minimum_profiles < 3:
        raise ValueError("production selection requires at least three profiles")
    return _select(
        candidates,
        eligibility="production_eligible",
        disposition_field="production_disposition",
        library_cap=library_cap,
        minimum_library_size=minimum_library_size,
        minimum_profiles=minimum_profiles,
        maximum_absolute_validation_spearman=maximum_absolute_validation_spearman,
        target_size=None,
    )


def select_size_matched_null(
    candidates: Iterable[HardAstScore],
    *,
    target_size: int,
    minimum_library_size: int,
    library_cap: int,
    minimum_profiles: int,
    maximum_absolute_validation_spearman: float,
) -> HardAstSelection:
    """Select exactly K executable nulls without efficacy or sign requirements."""
    if not 1 <= minimum_library_size <= target_size <= library_cap:
        raise ValueError(
            "size-matched null target_size must be in frozen admission bounds "
            f"[{minimum_library_size}, {library_cap}]"
        )
    if minimum_profiles < 3:
        raise ValueError("size-matched null requires at least three profiles")
    return _select(
        candidates,
        eligibility="null_eligible",
        disposition_field="null_disposition",
        library_cap=target_size,
        minimum_library_size=target_size,
        minimum_profiles=minimum_profiles,
        maximum_absolute_validation_spearman=maximum_absolute_validation_spearman,
        target_size=target_size,
    )


def select_size_matched_gp_control(
    candidates: Iterable[HardAstScore],
    *,
    target_size: int,
    minimum_library_size: int,
    library_cap: int,
    minimum_profiles: int,
    maximum_absolute_validation_spearman: float,
) -> HardAstSelection:
    """Select exactly K GP controls under the same real admission gates."""
    if not 1 <= minimum_library_size <= target_size <= library_cap:
        raise ValueError(
            "size-matched GP target_size must be in frozen admission bounds "
            f"[{minimum_library_size}, {library_cap}]"
        )
    return _select(
        candidates,
        eligibility="production_eligible",
        disposition_field="production_disposition",
        library_cap=target_size,
        minimum_library_size=target_size,
        minimum_profiles=minimum_profiles,
        maximum_absolute_validation_spearman=maximum_absolute_validation_spearman,
        target_size=target_size,
    )


def selected_library_effective_rank(
    selected: Sequence[HardAstScore], *, minimum_joint_rows: int = 2
) -> float:
    """Compute the frozen 2021 joint-support pooled-correlation entropy rank."""
    if not selected:
        raise ValueError("effective rank requires at least one selected factor")
    if minimum_joint_rows < 2:
        raise ValueError("minimum_joint_rows must be at least two")
    ids = [candidate.candidate_id for candidate in selected]
    if len(set(ids)) != len(ids):
        raise ValueError("effective rank factor IDs must be unique")
    frame = pd.concat(
        {candidate.candidate_id: candidate.validation_values for candidate in selected},
        axis=1,
    ).dropna()
    if len(frame) and not np.all(frame.index.get_level_values("datetime").year == 2021):
        raise ValueError("effective rank accepts only frozen 2021 validation rows")
    standardized: list[pd.DataFrame] = []
    for _, daily in frame.groupby(level="datetime", sort=False):
        values = daily.to_numpy(dtype=float)
        means = values.mean(axis=0)
        standard_deviations = values.std(axis=0, ddof=0)
        if np.any(~np.isfinite(standard_deviations)) or np.any(
            standard_deviations == 0.0
        ):
            continue
        standardized.append(
            pd.DataFrame(
                (values - means) / standard_deviations,
                index=daily.index,
                columns=daily.columns,
            )
        )
    if not standardized:
        raise ValueError(
            "effective rank has insufficient joint finite nonconstant rows"
        )
    pooled = pd.concat(standardized).to_numpy(dtype=float)
    if len(pooled) < minimum_joint_rows:
        raise ValueError("effective rank has insufficient joint rows")
    if pooled.shape[1] == 1:
        return 1.0
    correlation = np.corrcoef(pooled, rowvar=False)
    if correlation.shape != (pooled.shape[1], pooled.shape[1]) or not np.all(
        np.isfinite(correlation)
    ):
        raise ValueError("effective rank pooled correlation is not finite")
    eigenvalues = np.maximum(np.linalg.eigvalsh(correlation), 0.0)
    total = float(eigenvalues.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("effective rank eigenvalue mass is not positive")
    probabilities = eigenvalues / total
    positive = probabilities > 0.0
    entropy = -float(np.sum(probabilities[positive] * np.log(probabilities[positive])))
    return float(math.exp(entropy))


__all__ = [
    "HardAstCandidate",
    "HardAstScore",
    "HardAstScoringRun",
    "HardAstSelection",
    "score_hard_ast_candidates",
    "select_production_candidates",
    "select_size_matched_gp_control",
    "select_size_matched_null",
    "selected_library_effective_rank",
]
