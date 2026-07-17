from __future__ import annotations

import inspect
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from mirage_kan.dsl import AstNode
from mirage_kan.mining.e3 import PROFILE_SPECS


def _score(program: AstNode, context: object) -> float:
    del context
    return int(program.identity[:12], 16) / float(16**12)


def test_gp_generation_is_an_exact_deterministic_256_attempt_budget() -> None:
    from mirage_kan.mining.gp_control import generate_gp_attempts

    first = generate_gp_attempts(_score)
    second = generate_gp_attempts(_score)

    assert first.seed == 15485863
    assert first.attempts == second.attempts
    assert len(first.attempts) == 256
    assert Counter(attempt.profile for attempt in first.attempts) == {
        profile: 64 for profile in PROFILE_SPECS
    }
    for profile in PROFILE_SPECS:
        profile_attempts = [a for a in first.attempts if a.profile == profile]
        assert [a.attempt_index for a in profile_attempts] == list(range(64))
        assert all(a.phase == "initial" for a in profile_attempts[:16])
        assert all(a.phase in {"mutation", "crossover"} for a in profile_attempts[16:])


def test_initial_pairs_are_uniform_with_replacement_and_never_retried() -> None:
    from mirage_kan.mining.e3 import build_profile_atom_bank
    from mirage_kan.mining.gp_control import generate_gp_attempts

    result = generate_gp_attempts(_score)
    expected_rng = np.random.Generator(np.random.PCG64(15485863))

    for profile in PROFILE_SPECS:
        bank = build_profile_atom_bank(profile)
        profile_attempts = [a for a in result.attempts if a.profile == profile]
        for attempt in profile_attempts[:16]:
            positive = int(expected_rng.integers(len(bank)))
            negative = int(expected_rng.integers(len(bank)))
            assert attempt.positive_atom_index == positive
            assert attempt.negative_atom_index == negative
            if positive == negative:
                assert attempt.disposition == "invalid_same_atom"
                assert attempt.train_rank_ic is None

        # Consume the exact later-attempt random stream independently. The full
        # worked sequence is locked by the deterministic equality test above;
        # later semantics are asserted in the tournament test below.
        valid = [a for a in profile_attempts[:16] if a.disposition == "generated"]
        for attempt in profile_attempts[16:]:
            variation = (
                "mutation" if int(expected_rng.integers(2)) == 0 else "crossover"
            )
            assert attempt.phase == variation
            if not valid:
                assert attempt.disposition == "invalid_no_parent"
                continue
            tournament_count = 1 if variation == "mutation" else 2
            winners = []
            for _ in range(tournament_count):
                drawn = tuple(
                    valid[int(expected_rng.integers(len(valid)))].candidate_id
                    for _ in range(4)
                )
                assert attempt.tournaments[len(winners)] == drawn
                contestants = {item.candidate_id: item for item in valid}
                winner = min(
                    (contestants[candidate_id] for candidate_id in drawn),
                    key=lambda item: (
                        -abs(float(item.train_rank_ic)),
                        int(item.ast_nodes),
                        str(item.canonical_hash),
                    ),
                )
                winners.append(winner)
            if variation == "mutation":
                edge = int(expected_rng.integers(2))
                parent_atom = (
                    winners[0].positive_atom_index
                    if edge == 0
                    else winners[0].negative_atom_index
                )
                replacement = int(expected_rng.integers(len(bank) - 1))
                replacement += replacement >= parent_atom
                assert attempt.mutated_edge == ("positive" if edge == 0 else "negative")
                assert attempt.replacement_atom_index == replacement
            if attempt.disposition == "generated":
                valid.append(attempt)


def test_train_callback_has_no_validation_channel_and_only_scores_unique_valid_ast() -> (
    None
):
    from mirage_kan.mining.gp_control import generate_gp_attempts

    calls: list[tuple[AstNode, object]] = []

    def train_only(program: AstNode, context: object) -> float:
        calls.append((program, context))
        return 0.01

    result = generate_gp_attempts(train_only)
    scored = [a for a in result.attempts if a.train_rank_ic is not None]

    assert len(calls) == len(scored)
    assert all(call[0].op == "Sub" for call in calls)
    assert all(not hasattr(call[1], "validation") for call in calls)
    assert tuple(inspect.signature(generate_gp_attempts).parameters) == (
        "train_score",
        "seed",
        "attempts_per_profile",
    )


def test_duplicate_cancellation_and_no_parent_all_consume_without_hidden_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mirage_kan.mining.gp_control as gp
    from mirage_kan.mining.e3 import build_profile_atom_bank

    one_atom = build_profile_atom_bank("short_price")[:1]
    monkeypatch.setattr(gp, "PROFILE_SPECS", {"only": object()})
    monkeypatch.setattr(gp, "build_profile_atom_bank", lambda profile: one_atom)

    result = gp.generate_gp_attempts(_score)

    assert len(result.attempts) == 64
    assert [a.global_attempt_index for a in result.attempts] == list(range(64))
    assert all(a.disposition == "invalid_same_atom" for a in result.attempts[:16])
    assert all(a.disposition == "invalid_no_parent" for a in result.attempts[16:])


def test_gp_module_has_no_kan_model_or_torch_dependency() -> None:
    import mirage_kan.mining.gp_control as gp

    source = Path(gp.__file__).read_text()
    assert "CategoricalE3KAN" not in source
    assert "import torch" not in source
    assert "from torch" not in source


@pytest.mark.parametrize("attempts_per_profile", [-1, 0, 16, 63, 65])
def test_nonprotocol_attempt_budget_fails(attempts_per_profile: int) -> None:
    from mirage_kan.mining.gp_control import generate_gp_attempts

    with pytest.raises(ValueError, match="64"):
        generate_gp_attempts(_score, attempts_per_profile=attempts_per_profile)
