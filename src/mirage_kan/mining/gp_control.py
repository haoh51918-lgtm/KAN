"""Frozen typed GP/SR method control for the S2a v2 E3 experiment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from mirage_kan.dsl import AstNode
from mirage_kan.mining.e3 import PROFILE_SPECS, E3Atom, build_profile_atom_bank

GP_GENERATION_SEED = 15485863
ATTEMPTS_PER_PROFILE = 64
INITIAL_ATTEMPTS_PER_PROFILE = 16
TOURNAMENT_SIZE = 4


@dataclass(frozen=True)
class GpAttemptContext:
    """Train-only context exposed while one GP formula consumes its budget."""

    profile: str
    attempt_index: int
    global_attempt_index: int
    phase: str


@dataclass(frozen=True)
class GpAttempt:
    """Complete immutable ledger row for one budget-consuming GP attempt."""

    candidate_id: str
    profile: str
    attempt_index: int
    global_attempt_index: int
    phase: str
    disposition: str
    ast: AstNode | None
    canonical_hash: str | None
    ast_nodes: int | None
    positive_atom_index: int | None
    negative_atom_index: int | None
    positive_atom_hash: str | None
    negative_atom_hash: str | None
    parent_candidate_ids: tuple[str, ...]
    tournaments: tuple[tuple[str, ...], ...]
    mutated_edge: str | None
    replacement_atom_index: int | None
    train_rank_ic: float | None
    error: str | None = None

    def to_record(self) -> dict[str, object]:
        """Return a JSON-compatible record without dropping rejected proposals."""
        return {
            "candidate_id": self.candidate_id,
            "profile": self.profile,
            "attempt_index": self.attempt_index,
            "global_attempt_index": self.global_attempt_index,
            "phase": self.phase,
            "disposition": self.disposition,
            "ast": self.ast.to_dict() if self.ast is not None else None,
            "canonical_hash": self.canonical_hash,
            "ast_nodes": self.ast_nodes,
            "positive_atom_index": self.positive_atom_index,
            "negative_atom_index": self.negative_atom_index,
            "positive_atom_hash": self.positive_atom_hash,
            "negative_atom_hash": self.negative_atom_hash,
            "parent_candidate_ids": list(self.parent_candidate_ids),
            "tournaments": [list(tournament) for tournament in self.tournaments],
            "mutated_edge": self.mutated_edge,
            "replacement_atom_index": self.replacement_atom_index,
            "train_rank_ic": self.train_rank_ic,
            "error": self.error,
        }


@dataclass(frozen=True)
class GpGenerationResult:
    """Exact GP budget and the seed that determines every attempted formula."""

    seed: int
    attempts_per_profile: int
    attempts: tuple[GpAttempt, ...]


TrainScore = Callable[[AstNode, GpAttemptContext], float | None]


def _ast_nodes(ast: AstNode) -> int:
    return 1 + sum(_ast_nodes(child) for child in ast.children)


def _tournament(
    population: list[GpAttempt], rng: np.random.Generator
) -> tuple[GpAttempt, tuple[str, ...]]:
    drawn = tuple(
        population[int(rng.integers(len(population)))] for _ in range(TOURNAMENT_SIZE)
    )
    winner = min(
        drawn,
        key=lambda attempt: (
            -abs(float(attempt.train_rank_ic)),
            int(attempt.ast_nodes),
            str(attempt.canonical_hash),
        ),
    )
    return winner, tuple(attempt.candidate_id for attempt in drawn)


def _different_atom_index(
    rng: np.random.Generator, bank_size: int, excluded: int
) -> int:
    if bank_size < 2:
        raise ValueError("mutation requires at least two atoms")
    draw = int(rng.integers(bank_size - 1))
    return draw + int(draw >= excluded)


def _attempt(
    *,
    candidate_id: str,
    profile: str,
    attempt_index: int,
    global_attempt_index: int,
    phase: str,
    bank: tuple[E3Atom, ...],
    positive: int | None,
    negative: int | None,
    parents: tuple[GpAttempt, ...],
    tournaments: tuple[tuple[str, ...], ...],
    mutated_edge: str | None,
    replacement_atom_index: int | None,
    seen: set[str],
    train_score: TrainScore,
) -> GpAttempt:
    if positive is None or negative is None:
        return GpAttempt(
            candidate_id,
            profile,
            attempt_index,
            global_attempt_index,
            phase,
            "invalid_no_parent",
            None,
            None,
            None,
            positive,
            negative,
            None,
            None,
            tuple(parent.candidate_id for parent in parents),
            tournaments,
            mutated_edge,
            replacement_atom_index,
            None,
            "no prior valid individual is available",
        )

    ast = AstNode("Sub", (bank[positive].ast, bank[negative].ast))
    canonical_hash = ast.identity
    nodes = _ast_nodes(ast)
    common = {
        "candidate_id": candidate_id,
        "profile": profile,
        "attempt_index": attempt_index,
        "global_attempt_index": global_attempt_index,
        "phase": phase,
        "ast": ast,
        "canonical_hash": canonical_hash,
        "ast_nodes": nodes,
        "positive_atom_index": positive,
        "negative_atom_index": negative,
        "positive_atom_hash": bank[positive].canonical_hash,
        "negative_atom_hash": bank[negative].canonical_hash,
        "parent_candidate_ids": tuple(parent.candidate_id for parent in parents),
        "tournaments": tournaments,
        "mutated_edge": mutated_edge,
        "replacement_atom_index": replacement_atom_index,
    }
    if positive == negative:
        return GpAttempt(
            **common,
            disposition="invalid_same_atom",
            train_rank_ic=None,
            error="positive and negative edges select the same atom",
        )
    if canonical_hash in seen:
        return GpAttempt(
            **common,
            disposition="duplicate_formula",
            train_rank_ic=None,
            error="canonical formula was attempted previously in this profile",
        )
    seen.add(canonical_hash)
    context = GpAttemptContext(profile, attempt_index, global_attempt_index, phase)
    try:
        score = train_score(ast, context)
        score_value = float(score) if score is not None else float("nan")
    except (TypeError, ValueError, ArithmeticError) as error:
        return GpAttempt(
            **common,
            disposition="invalid_train_score",
            train_rank_ic=None,
            error=f"{type(error).__name__}: {error}",
        )
    if not math.isfinite(score_value):
        return GpAttempt(
            **common,
            disposition="invalid_train_score",
            train_rank_ic=None,
            error="train RankIC is absent or non-finite",
        )
    return GpAttempt(
        **common,
        disposition="generated",
        train_rank_ic=score_value,
        error=None,
    )


def generate_gp_attempts(
    train_score: TrainScore,
    *,
    seed: int = GP_GENERATION_SEED,
    attempts_per_profile: int = ATTEMPTS_PER_PROFILE,
) -> GpGenerationResult:
    """Run the exact v2 GP topology with no retry and no validation channel."""
    if not callable(train_score):
        raise TypeError("train_score must be callable")
    if attempts_per_profile != ATTEMPTS_PER_PROFILE:
        raise ValueError(
            "the frozen GP protocol requires exactly 64 attempts per profile"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    rng = np.random.Generator(np.random.PCG64(seed))
    attempts: list[GpAttempt] = []
    global_attempt_index = 0
    for profile in PROFILE_SPECS:
        bank = build_profile_atom_bank(profile)
        if not bank:
            raise RuntimeError(f"profile {profile!r} has an empty atom bank")
        population: list[GpAttempt] = []
        seen: set[str] = set()
        for attempt_index in range(attempts_per_profile):
            candidate_id = f"gp_{profile}_{attempt_index:03d}"
            parents: tuple[GpAttempt, ...] = ()
            tournaments: tuple[tuple[str, ...], ...] = ()
            mutated_edge = None
            replacement = None
            if attempt_index < INITIAL_ATTEMPTS_PER_PROFILE:
                phase = "initial"
                positive = int(rng.integers(len(bank)))
                negative = int(rng.integers(len(bank)))
            else:
                phase = "mutation" if int(rng.integers(2)) == 0 else "crossover"
                if not population:
                    positive = None
                    negative = None
                elif phase == "mutation":
                    parent, tournament = _tournament(population, rng)
                    parents = (parent,)
                    tournaments = (tournament,)
                    positive = int(parent.positive_atom_index)
                    negative = int(parent.negative_atom_index)
                    edge = int(rng.integers(2))
                    if edge == 0:
                        mutated_edge = "positive"
                        replacement = _different_atom_index(rng, len(bank), positive)
                        positive = replacement
                    else:
                        mutated_edge = "negative"
                        replacement = _different_atom_index(rng, len(bank), negative)
                        negative = replacement
                else:
                    first, first_tournament = _tournament(population, rng)
                    second, second_tournament = _tournament(population, rng)
                    parents = (first, second)
                    tournaments = (first_tournament, second_tournament)
                    positive = int(first.positive_atom_index)
                    negative = int(second.negative_atom_index)
            attempt = _attempt(
                candidate_id=candidate_id,
                profile=profile,
                attempt_index=attempt_index,
                global_attempt_index=global_attempt_index,
                phase=phase,
                bank=bank,
                positive=positive,
                negative=negative,
                parents=parents,
                tournaments=tournaments,
                mutated_edge=mutated_edge,
                replacement_atom_index=replacement,
                seen=seen,
                train_score=train_score,
            )
            attempts.append(attempt)
            if attempt.disposition == "generated":
                population.append(attempt)
            global_attempt_index += 1

    expected = attempts_per_profile * len(PROFILE_SPECS)
    if len(attempts) != expected:
        raise RuntimeError("GP generator did not consume the exact frozen budget")
    return GpGenerationResult(seed, attempts_per_profile, tuple(attempts))


__all__ = [
    "ATTEMPTS_PER_PROFILE",
    "GP_GENERATION_SEED",
    "GpAttempt",
    "GpAttemptContext",
    "GpGenerationResult",
    "generate_gp_attempts",
]
