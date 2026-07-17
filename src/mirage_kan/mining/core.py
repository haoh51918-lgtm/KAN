"""The complete label-screening seam for the frozen S2a typed miner."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from mirage_kan.data import PitPanel
from mirage_kan.dsl import AstNode, ProgramError, evaluate
from mirage_kan.dsl.core import WINDOWS

PROFILES = ("trend", "mean_reversion", "price_volume", "typed_composition")
FROZEN_WINDOWS = tuple(sorted(WINDOWS))
PRICE_LEAVES = ("Open", "High", "Low", "Close")
MAX_REQUIRED_LAG = 120


def ast_node_count(program: AstNode) -> int:
    """Count every operator and leaf in an AST."""
    return 1 + sum(ast_node_count(child) for child in program.children)


def ast_depth(program: AstNode) -> int:
    """Return one for a leaf and one plus the deepest child otherwise."""
    return 1 + max((ast_depth(child) for child in program.children), default=0)


def _series_sha256(series: pd.Series) -> str:
    digest = hashlib.sha256()
    digest.update(str(series.name).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(series, index=True).to_numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class MiningAttempt:
    """One budget-consuming generation attempt, successful or otherwise."""

    profile: str
    attempt_index: int
    candidate_id: str
    program: AstNode | None
    generation_disposition: str
    generation_error: str | None = None

    def to_record(self) -> dict[str, object]:
        program = self.program
        contract = program.validate() if program is not None else None
        return {
            "profile": self.profile,
            "attempt_index": self.attempt_index,
            "candidate_id": self.candidate_id,
            "generation_disposition": self.generation_disposition,
            "generation_error": self.generation_error,
            "ast": program.to_dict() if program is not None else None,
            "canonical_hash": program.identity if program is not None else None,
            "ast_depth": ast_depth(program) if program is not None else None,
            "ast_nodes": ast_node_count(program) if program is not None else None,
            "required_lookback": contract.lookback if contract is not None else None,
            "required_lag": contract.lookback - 1 if contract is not None else None,
        }


@dataclass(frozen=True)
class CandidateScore:
    """Explicit structural, support, label, and disposition record for one attempt."""

    candidate_id: str
    profile: str
    attempt_index: int
    program: AstNode | None
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
    eligible: bool
    disposition: str
    values: pd.Series = field(repr=False, compare=False)

    def to_record(self) -> dict[str, object]:
        record = {
            key: value
            for key, value in self.__dict__.items()
            if key not in {"program", "values"}
        } | {"ast": self.program.to_dict() if self.program is not None else None}
        contract = self.program.validate() if self.program is not None else None
        record["required_lookback"] = (
            contract.lookback if contract is not None else None
        )
        record["required_lag"] = contract.lookback - 1 if contract is not None else None
        return record


@dataclass(frozen=True)
class ScoringRun:
    """A fresh scoring pass bound to one exact observed or permuted label series."""

    candidates: tuple[CandidateScore, ...]
    label_mode: str
    label_sha256: str
    scoring_run_id: str
    train: tuple[str, str]
    validation: tuple[str, str]


@dataclass(frozen=True)
class SelectionResult:
    """Greedy selection plus an explicit terminal disposition for every candidate."""

    selected: tuple[CandidateScore, ...]
    by_candidate: dict[str, str]
    minimum_size_met: bool
    profile_quota_met: bool


def _leaf(name: str) -> AstNode:
    return AstNode(name)


def _window(rng: random.Random, *, short: bool = False) -> int:
    choices = FROZEN_WINDOWS[:5] if short else FROZEN_WINDOWS
    return choices[rng.randrange(len(choices))]


def _return(rng: random.Random) -> AstNode:
    return AstNode(
        "Return",
        (_leaf(PRICE_LEAVES[rng.randrange(len(PRICE_LEAVES))]),),
        {"window": _window(rng)},
    )


def _volume_change(rng: random.Random) -> AstNode:
    window = _window(rng)
    volume = _leaf("Volume")
    return AstNode(
        "SafeDiv",
        (
            AstNode("Delta", (volume,), {"window": window}),
            AstNode("Delay", (volume,), {"window": window}),
        ),
    )


def _trend_program(rng: random.Random) -> AstNode:
    price = _leaf(PRICE_LEAVES[rng.randrange(len(PRICE_LEAVES))])
    first = _window(rng)
    second = _window(rng)
    choice = rng.randrange(4)
    if choice == 0:
        return AstNode("Return", (price,), {"window": first})
    if choice == 1:
        return AstNode(
            "SafeDiv",
            (
                AstNode("Delta", (price,), {"window": first}),
                AstNode("Delay", (price,), {"window": first}),
            ),
        )
    if choice == 2:
        return AstNode(
            "SafeDiv",
            (
                AstNode(
                    "Sub",
                    (
                        AstNode("TsMean", (price,), {"window": first}),
                        AstNode("TsMean", (price,), {"window": second}),
                    ),
                ),
                AstNode("TsMean", (price,), {"window": second}),
            ),
        )
    return AstNode("TsMean", (_return(rng),), {"window": _window(rng, short=True)})


def _mean_reversion_program(rng: random.Random) -> AstNode:
    price = _leaf(PRICE_LEAVES[rng.randrange(len(PRICE_LEAVES))])
    window = _window(rng)
    mean = AstNode("TsMean", (price,), {"window": window})
    choice = rng.randrange(3)
    if choice == 0:
        return AstNode("SafeDiv", (AstNode("Sub", (mean, price)), price))
    if choice == 1:
        delayed = AstNode("Delay", (price,), {"window": window})
        return AstNode("SafeDiv", (AstNode("Sub", (delayed, price)), delayed))
    return AstNode(
        "CSRank", (AstNode("SafeDiv", (AstNode("Sub", (mean, price)), price)),)
    )


def _price_volume_program(rng: random.Random) -> AstNode:
    price_signal = _return(rng)
    volume_signal = _volume_change(rng)
    choice = rng.randrange(4)
    if choice == 0:
        return AstNode("Add", (price_signal, volume_signal))
    if choice == 1:
        return AstNode("Sub", (price_signal, volume_signal))
    if choice == 2:
        ratio = AstNode(
            "SafeDiv",
            (
                _leaf("Volume"),
                AstNode("TsMean", (_leaf("Volume"),), {"window": _window(rng)}),
            ),
        )
        return AstNode("Sub", (ratio, AstNode("Constant", params={"value": 1.0})))
    return AstNode("CSRank", (AstNode("Add", (price_signal, volume_signal)),))


def _typed_composition_program(rng: random.Random) -> AstNode:
    left = _return(rng) if rng.randrange(2) == 0 else _volume_change(rng)
    right = _return(rng) if rng.randrange(2) == 0 else _volume_change(rng)
    combined = AstNode("Add" if rng.randrange(2) == 0 else "Sub", (left, right))
    choice = rng.randrange(4)
    if choice == 0:
        return combined
    if choice == 1:
        return AstNode("TsMean", (combined,), {"window": _window(rng, short=True)})
    if choice == 2:
        return AstNode("Delay", (combined,), {"window": _window(rng)})
    return AstNode("CSRank", (combined,))


def generate_attempts(
    *, seed: int, attempts_per_profile: int, profiles: Sequence[str] = PROFILES
) -> tuple[MiningAttempt, ...]:
    """Generate exactly the requested attempt budget with no hidden retries."""
    if attempts_per_profile < 0:
        raise ValueError("attempts_per_profile must be nonnegative")
    unknown = sorted(set(profiles).difference(PROFILES))
    if unknown:
        raise ValueError(f"unknown miner profiles: {unknown}")
    builders = {
        "trend": _trend_program,
        "mean_reversion": _mean_reversion_program,
        "price_volume": _price_volume_program,
        "typed_composition": _typed_composition_program,
    }
    rng = random.Random(seed)
    attempts: list[MiningAttempt] = []
    for profile in profiles:
        for attempt_index in range(attempts_per_profile):
            candidate_id = f"{profile}_{attempt_index:03d}"
            try:
                program = builders[profile](rng)
                contract = program.validate()
                if ast_depth(program) > 6 or ast_node_count(program) > 20:
                    raise ProgramError("generated AST exceeds frozen S2a limits")
                if contract.lookback > MAX_REQUIRED_LAG + 1:
                    raise ProgramError(
                        "generated AST required lag exceeds frozen S2a maximum of "
                        f"{MAX_REQUIRED_LAG}: lookback={contract.lookback}"
                    )
            except (ProgramError, ValueError) as error:
                attempts.append(
                    MiningAttempt(
                        profile,
                        attempt_index,
                        candidate_id,
                        None,
                        "invalid_generation",
                        str(error),
                    )
                )
            else:
                attempts.append(
                    MiningAttempt(
                        profile, attempt_index, candidate_id, program, "generated"
                    )
                )
    return tuple(attempts)


def permute_labels_within_date(labels: pd.Series, *, seed: int) -> pd.Series:
    """Deterministically permute finite labels independently within every date."""
    if (
        not isinstance(labels.index, pd.MultiIndex)
        or "datetime" not in labels.index.names
    ):
        raise ValueError("labels must use a MultiIndex with a datetime level")
    rng = np.random.default_rng(seed)
    permuted = labels.copy()
    dates = labels.index.get_level_values("datetime")
    for date in pd.Index(dates).unique():
        positions = np.flatnonzero(dates == date)
        finite_positions = positions[
            np.isfinite(labels.iloc[positions].to_numpy(dtype=float))
        ]
        if len(finite_positions) > 1:
            values = labels.iloc[finite_positions].to_numpy(copy=True)
            permuted.iloc[finite_positions] = values[rng.permutation(len(values))]
    return permuted.rename(labels.name)


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


def score_attempts(
    attempts: Iterable[MiningAttempt],
    panel: PitPanel,
    labels: pd.Series,
    *,
    train: tuple[str, str],
    validation: tuple[str, str],
    minimum_coverage: float,
    minimum_absolute_rank_ic: float,
    label_mode: str,
) -> ScoringRun:
    """Execute and freshly score attempts using train/validation labels only."""
    attempts = tuple(attempts)
    dates = pd.to_datetime(panel.raw.index.get_level_values("datetime"))
    validation_end = pd.Timestamp(validation[1])
    if dates.max() > validation_end:
        raise ValueError("candidate scoring panel crosses the validation boundary")
    labels = labels.reindex(panel.raw.index)
    if labels.name != "fwd":
        raise ValueError("candidate scoring requires the exact fwd label")
    label_hash = _series_sha256(labels)
    run_payload = {
        "label_mode": label_mode,
        "label_sha256": label_hash,
        "train": train,
        "validation": validation,
        "attempts": [attempt.to_record() for attempt in attempts],
    }
    scoring_run_id = hashlib.sha256(
        json.dumps(run_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    seen: set[str] = set()
    scores: list[CandidateScore] = []
    train_mask = _date_mask(panel.raw.index, train)
    validation_mask = _date_mask(panel.raw.index, validation)
    development_mask = train_mask | validation_mask
    label_finite = pd.Series(
        np.isfinite(labels.to_numpy(dtype=float)), index=panel.raw.index
    )
    denominator = panel.membership & label_finite & development_mask

    for attempt in attempts:
        empty = pd.Series(np.nan, index=panel.raw.index, dtype=float)
        if attempt.program is None:
            scores.append(
                CandidateScore(
                    attempt.candidate_id,
                    attempt.profile,
                    attempt.attempt_index,
                    None,
                    None,
                    None,
                    None,
                    None,
                    False,
                    False,
                    0,
                    int(denominator.sum()),
                    0.0,
                    None,
                    None,
                    False,
                    None,
                    False,
                    "invalid_generation",
                    empty,
                )
            )
            continue
        program = attempt.program
        canonical_hash = program.identity
        contract = program.validate()
        unique = canonical_hash not in seen
        seen.add(canonical_hash)
        result = evaluate(program, panel)
        effective = result.support & panel.membership & label_finite
        support_rows = int((effective & development_mask).sum())
        eligible_rows = int(denominator.sum())
        coverage = support_rows / eligible_rows if eligible_rows else 0.0
        values = result.values.where(effective)
        train_ic = _mean_daily_spearman(
            values.where(train_mask), labels.where(train_mask)
        )
        validation_ic = _mean_daily_spearman(
            values.where(validation_mask), labels.where(validation_mask)
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
        if not unique:
            disposition = "duplicate_canonical_identity"
        elif coverage < minimum_coverage:
            disposition = "ineligible_coverage"
        elif minimum_score is None or minimum_score < minimum_absolute_rank_ic:
            disposition = "ineligible_rank_ic"
        elif not sign_agreement:
            disposition = "ineligible_sign_disagreement"
        else:
            disposition = "eligible"
        scores.append(
            CandidateScore(
                candidate_id=attempt.candidate_id,
                profile=attempt.profile,
                attempt_index=attempt.attempt_index,
                program=program,
                canonical_hash=canonical_hash,
                ast_depth=ast_depth(program),
                ast_nodes=ast_node_count(program),
                output_type=contract.output_type.value,
                causal=contract.causal,
                unique=unique,
                support_rows=support_rows,
                eligible_rows=eligible_rows,
                coverage=coverage,
                train_rank_ic=train_ic,
                validation_rank_ic=validation_ic,
                sign_agreement=sign_agreement,
                minimum_score=minimum_score,
                eligible=disposition == "eligible",
                disposition=disposition,
                values=values.loc[validation_mask],
            )
        )
    return ScoringRun(
        tuple(scores), label_mode, label_hash, scoring_run_id, train, validation
    )


def _ordered(candidates: Iterable[CandidateScore]) -> list[CandidateScore]:
    return sorted(
        (candidate for candidate in candidates if candidate.eligible),
        key=lambda candidate: (
            -float(candidate.minimum_score),
            int(candidate.ast_nodes),
            str(candidate.canonical_hash),
        ),
    )


def _diversity_disposition(
    candidate: CandidateScore,
    selected: Sequence[CandidateScore],
    maximum_absolute_spearman: float,
) -> str | None:
    for existing in selected:
        correlation = _mean_daily_spearman(candidate.values, existing.values)
        if correlation is None:
            return "rejected_diversity_undefined"
        magnitude = abs(correlation)
        if magnitude >= maximum_absolute_spearman or np.isclose(
            magnitude, maximum_absolute_spearman, rtol=0.0, atol=1e-12
        ):
            return "rejected_diversity"
    return None


def greedy_select(
    candidates: Iterable[CandidateScore],
    *,
    library_cap: int,
    minimum_library_size: int,
    minimum_profiles: int,
    maximum_absolute_spearman: float,
) -> SelectionResult:
    """Apply frozen ordering, prospective diversity, and profile representation."""
    all_candidates = list(candidates)
    ordered = _ordered(all_candidates)
    selected: list[CandidateScore] = []
    dispositions = {
        candidate.candidate_id: candidate.disposition for candidate in all_candidates
    }
    seen_profiles: set[str] = set()
    deferred: list[CandidateScore] = []
    for candidate in ordered:
        if len(selected) >= library_cap:
            dispositions[candidate.candidate_id] = "eligible_not_selected_cap"
            continue
        if len(seen_profiles) < minimum_profiles and candidate.profile in seen_profiles:
            deferred.append(candidate)
            continue
        rejection = _diversity_disposition(
            candidate, selected, maximum_absolute_spearman
        )
        if rejection is not None:
            dispositions[candidate.candidate_id] = rejection
            continue
        selected.append(candidate)
        seen_profiles.add(candidate.profile)
        dispositions[candidate.candidate_id] = "selected"
    for candidate in deferred:
        if dispositions[candidate.candidate_id] != "eligible":
            continue
        if len(selected) >= library_cap:
            dispositions[candidate.candidate_id] = "eligible_not_selected_cap"
        elif (
            rejection := _diversity_disposition(
                candidate, selected, maximum_absolute_spearman
            )
        ) is not None:
            dispositions[candidate.candidate_id] = rejection
        else:
            selected.append(candidate)
            seen_profiles.add(candidate.profile)
            dispositions[candidate.candidate_id] = "selected"
    return SelectionResult(
        tuple(selected),
        dispositions,
        len(selected) >= minimum_library_size,
        len(seen_profiles) >= minimum_profiles,
    )


def select_random_control(
    attempts: Iterable[MiningAttempt],
    panel: PitPanel,
    *,
    seed: int,
    library_cap: int,
    minimum_coverage: float,
    period: tuple[str, str] | None = None,
) -> tuple[CandidateScore, ...]:
    """Sample valid unique typed candidates without consulting any label metric."""
    seen: set[str] = set()
    candidates: list[CandidateScore] = []
    period_mask = (
        pd.Series(_date_mask(panel.raw.index, period), index=panel.raw.index)
        if period is not None
        else pd.Series(True, index=panel.raw.index)
    )
    membership = panel.membership & period_mask
    membership_rows = int(membership.sum())
    for attempt in attempts:
        if attempt.program is None:
            continue
        program = attempt.program
        canonical_hash = program.identity
        if canonical_hash in seen:
            continue
        seen.add(canonical_hash)
        contract = program.validate()
        result = evaluate(program, panel)
        support = result.support & membership
        support_rows = int(support.sum())
        coverage = support_rows / membership_rows if membership_rows else 0.0
        if coverage < minimum_coverage:
            continue
        candidates.append(
            CandidateScore(
                candidate_id=attempt.candidate_id,
                profile=attempt.profile,
                attempt_index=attempt.attempt_index,
                program=program,
                canonical_hash=canonical_hash,
                ast_depth=ast_depth(program),
                ast_nodes=ast_node_count(program),
                output_type=contract.output_type.value,
                causal=contract.causal,
                unique=True,
                support_rows=support_rows,
                eligible_rows=membership_rows,
                coverage=coverage,
                train_rank_ic=None,
                validation_rank_ic=None,
                sign_agreement=False,
                minimum_score=None,
                eligible=True,
                disposition="random_label_free_eligible",
                values=pd.Series(dtype=float),
            )
        )
    candidates.sort(key=lambda candidate: str(candidate.canonical_hash))
    random.Random(seed).shuffle(candidates)
    return tuple(candidates[:library_cap])
