from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from mirage_kan.data import PitPanel
from mirage_kan.dsl import AstNode, evaluate
from mirage_kan.mining.e3 import build_profile_atom_bank


TRAIN = ("2020-01-01", "2020-03-31")
VALIDATION = ("2021-01-01", "2021-03-31")


def _panel() -> PitPanel:
    dates = pd.DatetimeIndex(
        [*pd.date_range(*TRAIN, freq="D"), *pd.date_range(*VALIDATION, freq="D")]
    )
    rows = []
    for date_index, date in enumerate(dates):
        for instrument_index, instrument in enumerate("ABCDEF"):
            growth = 1.0 + (instrument_index - 2.5) * 0.0015
            close = (10.0 + instrument_index) * growth**date_index
            rows.append(
                {
                    "datetime": date,
                    "instrument": instrument,
                    "open": close * (1.0 + 0.0002 * instrument_index),
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1000.0 + date_index * (instrument_index + 1),
                    "in_universe": True,
                }
            )
    return PitPanel.from_frame(pd.DataFrame(rows))


def _strict_ast(profile: str, left: int = 0, right: int = 1) -> AstNode:
    bank = build_profile_atom_bank(profile)
    return AstNode("Sub", (bank[left].ast, bank[right].ast))


def _labels(panel: PitPanel, ast: AstNode) -> pd.Series:
    result = evaluate(ast, panel)
    values = result.values.where(result.support & panel.membership)
    return values.rename("fwd")


def _score_candidates(candidates: list[object], labels: pd.Series):
    from mirage_kan.mining.v2_scoring import score_hard_ast_candidates

    return score_hard_ast_candidates(
        candidates,
        _panel(),
        labels,
        train=TRAIN,
        validation=VALIDATION,
        minimum_coverage=0.85,
        minimum_absolute_train_rank_ic=0.005,
        minimum_absolute_validation_rank_ic=0.005,
        minimum_soft_hard_pearson=0.98,
        maximum_soft_hard_nrmse=0.10,
        minimum_gate_probability_margin=0.05,
        label_mode="observed",
    )


def test_common_scoring_separates_real_admission_from_size_matched_null() -> None:
    from mirage_kan.mining.v2_scoring import HardAstCandidate

    panel = _panel()
    ast = _strict_ast("short_price")
    labels = _labels(panel, ast)
    candidate = HardAstCandidate(
        "gp_0", "short_price", 0, ast, "typed_gp_sr", lineage_complete=True
    )
    observed = _score_candidates([candidate], labels).candidates[0]

    reversed_labels = labels.copy()
    validation_mask = reversed_labels.index.get_level_values(
        "datetime"
    ) >= pd.Timestamp(VALIDATION[0])
    reversed_labels.loc[validation_mask] *= -1.0
    null = _score_candidates([candidate], reversed_labels).candidates[0]

    assert observed.production_eligible is True
    assert observed.null_eligible is True
    assert null.production_eligible is False
    assert null.production_disposition in {
        "ineligible_validation_rank_ic",
        "ineligible_sign_disagreement",
    }
    assert null.null_eligible is True
    assert null.null_disposition == "null_eligible"


def test_kan_fidelity_lineage_and_strict_ast_gates_are_not_approximated() -> None:
    from mirage_kan.mining.v2_scoring import HardAstCandidate

    panel = _panel()
    ast = _strict_ast("reversal")
    hard = evaluate(ast, panel).values
    labels = _labels(panel, ast)
    bad_lineage_ast = _strict_ast("reversal", 2, 3)
    low_margin_ast = _strict_ast("reversal", 4, 5)
    candidates = [
        HardAstCandidate(
            "good",
            "reversal",
            0,
            ast,
            "kan_e3",
            lineage_complete=True,
            soft_values=hard.copy(),
            gate_probability_margin=0.08,
        ),
        HardAstCandidate(
            "bad_lineage",
            "reversal",
            1,
            bad_lineage_ast,
            "kan_e3",
            lineage_complete=False,
            soft_values=evaluate(bad_lineage_ast, panel).values,
            gate_probability_margin=0.08,
        ),
        HardAstCandidate(
            "not_strict",
            "reversal",
            2,
            AstNode("Return", (AstNode("Close"),), {"window": 2}),
            "kan_e3",
            lineage_complete=True,
            soft_values=hard.copy(),
            gate_probability_margin=0.08,
        ),
        HardAstCandidate(
            "low_margin",
            "reversal",
            3,
            low_margin_ast,
            "kan_e3",
            lineage_complete=True,
            soft_values=evaluate(low_margin_ast, panel).values,
            gate_probability_margin=0.01,
        ),
    ]
    scores = _score_candidates(candidates, labels).candidates

    assert scores[0].fidelity_pearson == pytest.approx(1.0)
    assert scores[0].fidelity_nrmse == pytest.approx(0.0)
    assert scores[0].production_eligible is True
    assert scores[1].production_disposition == "ineligible_lineage"
    assert scores[1].null_eligible is True
    assert scores[2].production_disposition == "invalid_strict_ast"
    assert scores[3].production_disposition == "ineligible_gate_margin"
    assert scores[3].null_eligible is True


def test_ineligible_early_duplicate_cannot_block_a_complete_kan_lineage() -> None:
    from mirage_kan.mining.v2_scoring import HardAstCandidate

    panel = _panel()
    ast = _strict_ast("short_price")
    hard = evaluate(ast, panel).values
    labels = _labels(panel, ast)
    scores = _score_candidates(
        [
            HardAstCandidate(
                "early_bad",
                "short_price",
                0,
                ast,
                "kan_e3",
                lineage_complete=False,
                soft_values=hard,
                gate_probability_margin=0.08,
            ),
            HardAstCandidate(
                "later_complete",
                "short_price",
                1,
                ast,
                "kan_e3",
                lineage_complete=True,
                soft_values=hard,
                gate_probability_margin=0.08,
            ),
        ],
        labels,
    ).candidates

    assert scores[0].unique is False
    assert scores[0].production_disposition == "duplicate_canonical_identity"
    assert scores[1].unique is True
    assert scores[1].production_eligible is True


def _selection_score(
    candidate_id: str,
    profile: str,
    score: float,
    pattern: list[float],
    *,
    canonical_hash: str | None = None,
):
    from mirage_kan.mining.v2_scoring import HardAstScore

    index = pd.MultiIndex.from_product(
        [pd.date_range("2021-01-04", periods=3), list("ABCDEF")],
        names=["datetime", "instrument"],
    )
    values = pd.Series(np.tile(pattern, 3), index=index, dtype=float)
    ast = _strict_ast(profile)
    return HardAstScore(
        candidate_id=candidate_id,
        profile=profile,
        attempt_index=0,
        method="typed_gp_sr",
        ast=ast,
        canonical_hash=canonical_hash or candidate_id.rjust(64, "0")[-64:],
        ast_depth=3,
        ast_nodes=7,
        output_type="dimensionless_ts",
        causal=True,
        unique=True,
        support_rows=len(index),
        eligible_rows=len(index),
        coverage=1.0,
        train_rank_ic=score,
        validation_rank_ic=score,
        sign_agreement=True,
        minimum_score=score,
        fidelity_pearson=1.0,
        fidelity_nrmse=0.0,
        gate_probability_margin=None,
        fidelity_gate_met=True,
        lineage_gate_met=True,
        production_eligible=True,
        null_eligible=True,
        production_disposition="production_eligible",
        null_disposition="null_eligible",
        validation_values=values,
    )


def test_selection_enforces_profile_diversity_ties_and_terminal_dispositions() -> None:
    from mirage_kan.mining.v2_scoring import select_production_candidates

    patterns = [
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 5, 4, 3],
        [0, 1, 4, 2, 5, 3],
        [0, 1, 4, 3, 2, 5],
        [0, 1, 4, 5, 2, 3],
        [0, 2, 3, 4, 5, 1],
        [0, 2, 5, 1, 3, 4],
        [0, 2, 5, 4, 3, 1],
    ]
    candidates = [
        _selection_score(
            "b", "short_price", 0.02, patterns[0], canonical_hash="b" * 64
        ),
        _selection_score(
            "a", "short_price", 0.02, patterns[1], canonical_hash="a" * 64
        ),
        _selection_score("c", "long_price", 0.019, patterns[2]),
        _selection_score("d", "reversal", 0.018, patterns[3]),
        _selection_score("e", "price_volume", 0.0175, patterns[4]),
        _selection_score("f", "short_price", 0.017, patterns[5]),
        _selection_score("g", "long_price", 0.016, patterns[6]),
        _selection_score("h", "reversal", 0.015, patterns[7]),
        replace(
            _selection_score("duplicate_signal", "price_volume", 0.014, patterns[0]),
            validation_values=_selection_score(
                "a", "short_price", 0.02, patterns[1]
            ).validation_values,
        ),
    ]
    result = select_production_candidates(
        candidates,
        library_cap=9,
        minimum_library_size=8,
        minimum_profiles=3,
        maximum_absolute_validation_spearman=0.80,
    )

    assert result.complete is True
    assert [item.candidate_id for item in result.selected][:3] == ["a", "c", "d"]
    assert result.dispositions["a"] == "selected"
    assert result.dispositions["b"] in {"selected", "eligible_not_selected_cap"}
    assert result.dispositions["duplicate_signal"] == "rejected_diversity"
    result.require_complete()


def test_size_matched_null_requires_exact_k_and_three_profiles() -> None:
    from mirage_kan.mining.v2_scoring import select_size_matched_null

    patterns = [
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 5, 4, 3],
        [0, 1, 4, 2, 5, 3],
        [0, 1, 4, 3, 2, 5],
        [0, 1, 4, 5, 2, 3],
        [0, 2, 3, 4, 5, 1],
        [0, 2, 5, 1, 3, 4],
        [0, 2, 5, 4, 3, 1],
    ]
    candidates = [
        _selection_score(
            chr(ord("a") + index),
            ("short_price", "long_price", "reversal", "price_volume")[index % 4],
            0.01 - index * 0.0001,
            pattern,
        )
        for index, pattern in enumerate(patterns)
    ]
    complete = select_size_matched_null(
        candidates,
        target_size=8,
        minimum_library_size=8,
        library_cap=16,
        minimum_profiles=3,
        maximum_absolute_validation_spearman=0.80,
    )
    assert complete.complete is True
    assert complete.exact_size_met is True
    assert len(complete.selected) == 8

    insufficient = select_size_matched_null(
        candidates[:7],
        target_size=8,
        minimum_library_size=8,
        library_cap=16,
        minimum_profiles=3,
        maximum_absolute_validation_spearman=0.80,
    )
    assert insufficient.complete is False
    assert insufficient.exact_size_met is False
    assert insufficient.profile_quota_met is True
    with pytest.raises(RuntimeError, match="incomplete"):
        insufficient.require_complete()


def test_size_matched_gp_control_requires_exact_production_k() -> None:
    from mirage_kan.mining.v2_scoring import select_size_matched_gp_control

    patterns = [
        [0, 1, 2, 3, 4, 5],
        [0, 2, 4, 1, 5, 3],
        [3, 0, 5, 2, 1, 4],
        [1, 4, 0, 5, 3, 2],
        [2, 5, 1, 4, 0, 3],
        [4, 1, 3, 0, 5, 2],
        [5, 3, 0, 2, 4, 1],
        [2, 0, 4, 5, 1, 3],
    ]
    profiles = ["short_price", "long_price", "reversal", "price_volume"]
    candidates = [
        _selection_score(
            f"gp_{index}", profiles[index % 4], 0.02 - index * 0.0001, patterns[index]
        )
        for index in range(8)
    ]
    complete = select_size_matched_gp_control(
        candidates,
        target_size=8,
        minimum_library_size=8,
        library_cap=16,
        minimum_profiles=3,
        maximum_absolute_validation_spearman=1.0,
    )
    assert complete.complete is True
    assert len(complete.selected) == 8

    insufficient = select_size_matched_gp_control(
        candidates[:7],
        target_size=8,
        minimum_library_size=8,
        library_cap=16,
        minimum_profiles=3,
        maximum_absolute_validation_spearman=1.0,
    )
    assert insufficient.complete is False
    with pytest.raises(RuntimeError, match="incomplete"):
        insufficient.require_complete()


def test_v5_six_factor_selection_bounds_accept_six_and_reject_five_or_seventeen() -> (
    None
):
    from itertools import islice, permutations

    from mirage_kan.mining.v2_scoring import (
        select_production_candidates,
        select_size_matched_gp_control,
        select_size_matched_null,
    )

    patterns = tuple(list(pattern) for pattern in islice(permutations(range(6)), 17))
    profiles = ("short_price", "long_price", "reversal", "price_volume")
    candidates = tuple(
        _selection_score(
            f"v4_{index}",
            profiles[index % len(profiles)],
            0.02 - index * 0.001,
            pattern,
        )
        for index, pattern in enumerate(patterns)
    )

    production = select_production_candidates(
        candidates[:6],
        library_cap=16,
        minimum_library_size=6,
        minimum_profiles=3,
        maximum_absolute_validation_spearman=1.0,
    )
    null = select_size_matched_null(
        candidates[:6],
        target_size=6,
        minimum_library_size=6,
        library_cap=16,
        minimum_profiles=3,
        maximum_absolute_validation_spearman=1.0,
    )
    gp = select_size_matched_gp_control(
        candidates[:6],
        target_size=6,
        minimum_library_size=6,
        library_cap=16,
        minimum_profiles=3,
        maximum_absolute_validation_spearman=1.0,
    )
    assert len(production.selected) == len(null.selected) == len(gp.selected) == 6
    assert production.complete and null.complete and gp.complete

    undersized = select_production_candidates(
        candidates[:5],
        library_cap=16,
        minimum_library_size=6,
        minimum_profiles=3,
        maximum_absolute_validation_spearman=1.0,
    )
    assert not undersized.complete
    with pytest.raises(RuntimeError, match="incomplete"):
        undersized.require_complete()
    oversized = select_production_candidates(
        candidates,
        library_cap=16,
        minimum_library_size=6,
        minimum_profiles=3,
        maximum_absolute_validation_spearman=1.0,
    )
    assert len(oversized.selected) == 16
    assert (
        tuple(oversized.dispositions.values()).count("eligible_not_selected_cap") == 1
    )

    for invalid in (5, 17):
        with pytest.raises(ValueError, match="6|16"):
            select_size_matched_null(
                candidates,
                target_size=invalid,
                minimum_library_size=6,
                library_cap=16,
                minimum_profiles=3,
                maximum_absolute_validation_spearman=1.0,
            )
        with pytest.raises(ValueError, match="6|16"):
            select_size_matched_gp_control(
                candidates,
                target_size=invalid,
                minimum_library_size=6,
                library_cap=16,
                minimum_profiles=3,
                maximum_absolute_validation_spearman=1.0,
            )


def test_effective_rank_values_do_not_depend_on_missing_labels() -> None:
    from mirage_kan.mining.v2_scoring import (
        HardAstCandidate,
        selected_library_effective_rank,
    )

    panel = _panel()
    first_ast = _strict_ast("short_price")
    second_ast = _strict_ast("long_price")
    candidates = [
        HardAstCandidate("a", "short_price", 0, first_ast, "typed_gp_sr", True),
        HardAstCandidate("b", "long_price", 1, second_ast, "typed_gp_sr", True),
    ]
    complete_labels = _labels(panel, first_ast)
    missing_labels = complete_labels.copy()
    validation_rows = missing_labels.index.get_level_values("datetime") >= pd.Timestamp(
        VALIDATION[0]
    )
    missing_labels.loc[
        validation_rows & (missing_labels.index.get_level_values("instrument") == "F")
    ] = np.nan

    complete = _score_candidates(candidates, complete_labels).candidates
    missing = _score_candidates(candidates, missing_labels).candidates

    pd.testing.assert_series_equal(
        complete[0].validation_values, missing[0].validation_values
    )
    pd.testing.assert_series_equal(
        complete[1].validation_values, missing[1].validation_values
    )
    assert selected_library_effective_rank(complete) == pytest.approx(
        selected_library_effective_rank(missing), abs=1e-12
    )


def test_effective_rank_uses_joint_2021_rows_population_zscore_and_entropy() -> None:
    from mirage_kan.mining.v2_scoring import selected_library_effective_rank

    first = _selection_score("a", "short_price", 0.01, [1, -1, 1, -1, 1, -1])
    second = _selection_score("b", "long_price", 0.01, [1, 1, -1, -1, 0, 0])
    independent = selected_library_effective_rank([first, second])
    duplicate = selected_library_effective_rank(
        [first, replace(second, validation_values=first.validation_values.copy())]
    )

    assert independent == pytest.approx(2.0, abs=1e-12)
    assert duplicate == pytest.approx(1.0, abs=1e-12)


def test_illegal_scoring_and_effective_rank_inputs_fail_closed() -> None:
    from mirage_kan.mining.v2_scoring import (
        HardAstCandidate,
        score_hard_ast_candidates,
        selected_library_effective_rank,
    )

    panel = _panel()
    ast = _strict_ast("short_price")
    labels = _labels(panel, ast)
    with pytest.raises(ValueError, match="method"):
        HardAstCandidate("x", "short_price", 0, ast, "random", True)
    with pytest.raises(ValueError, match="validation boundar"):
        score_hard_ast_candidates(
            [HardAstCandidate("x", "short_price", 0, ast, "typed_gp_sr", True)],
            panel,
            labels,
            train=TRAIN,
            validation=("2021-01-01", "2020-12-31"),
            minimum_coverage=0.85,
            minimum_absolute_train_rank_ic=0.005,
            minimum_absolute_validation_rank_ic=0.005,
            minimum_soft_hard_pearson=0.98,
            maximum_soft_hard_nrmse=0.1,
            minimum_gate_probability_margin=0.05,
            label_mode="observed",
        )
    constant = replace(
        _selection_score("a", "short_price", 0.01, [1, 2, 3, 4, 5, 6]),
        validation_values=pd.Series(
            1.0,
            index=_selection_score(
                "a", "short_price", 0.01, [1, 2, 3, 4, 5, 6]
            ).validation_values.index,
        ),
    )
    with pytest.raises(ValueError, match="joint"):
        selected_library_effective_rank([constant])
