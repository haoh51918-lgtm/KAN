from __future__ import annotations

import numpy as np
import pytest

from mirage_kan.dsl import AstNode, DslType, ProgramError, evaluate


def leaf(name: str) -> AstNode:
    return AstNode(name)


def test_operator_contract_and_window_support(tiny_panel) -> None:
    program = AstNode("Return", (leaf("Close"),), {"window": 2})
    contract = program.contract()
    assert contract.output_type is DslType.DIMENSIONLESS_TS
    assert contract.lookback == 3
    assert contract.causal is True
    assert contract.domain == "lagged denominator must be finite and nonzero"
    assert contract.mask_rule == "current and lagged child support"

    result = evaluate(program, tiny_panel)
    assert not bool(result.support.loc[("2020-01-01", "A")])
    assert not bool(result.support.loc[("2020-01-02", "A")])
    assert not bool(result.support.loc[("2020-01-03", "A")])
    assert bool(result.support.loc[("2020-01-04", "A")])
    assert not bool(result.support.loc[("2020-01-05", "A")])
    assert np.isfinite(result.values[result.support]).all()


def test_invalid_types_windows_and_future_access_are_rejected() -> None:
    with pytest.raises(ProgramError, match="type"):
        AstNode("Sub", (leaf("Close"), leaf("Volume"))).validate()
    with pytest.raises(ProgramError, match="window"):
        AstNode("Delay", (leaf("Close"),), {"window": -1}).validate()
    with pytest.raises(ProgramError, match="unknown operator"):
        AstNode("Future", (leaf("Close"),), {"window": 1}).validate()


def test_canonicalization_preserves_semantics() -> None:
    left = AstNode("Add", (AstNode("Return", (leaf("Close"),), {"window": 2}), AstNode("Return", (leaf("Open"),), {"window": 2})))
    right = AstNode("Add", tuple(reversed(left.children)))
    assert left.canonical_json() == right.canonical_json()
    assert left.identity == right.identity

    pos = AstNode("Sub", left.children)
    neg = AstNode("Sub", tuple(reversed(left.children)))
    assert pos.identity != neg.identity


def test_seed_program_evaluates_deterministically(tiny_panel) -> None:
    program = AstNode("SafeDiv", (AstNode("Sub", (leaf("High"), leaf("Low"))), leaf("Close")))
    first = evaluate(program, tiny_panel)
    second = evaluate(AstNode.from_dict(program.to_dict()), tiny_panel)
    np.testing.assert_allclose(first.values, second.values, equal_nan=True)
    assert first.support.equals(second.support)


def test_scalar_constant_broadcasts_without_conflating_membership(tiny_panel) -> None:
    program = AstNode(
        "Sub",
        (
            AstNode(
                "SafeDiv",
                (
                    AstNode("Volume"),
                    AstNode("TsMean", (AstNode("Volume"),), {"window": 2}),
                ),
            ),
            AstNode("Constant", params={"value": 1.0}),
        ),
    )
    result = evaluate(program, tiny_panel)
    expected = [
        np.nan,
        np.nan,
        101 / 100.5 - 1,
        101 / 100.5 - 1,
        102 / 101.5 - 1,
        102 / 101.5 - 1,
        103 / 102.5 - 1,
        103 / 102.5 - 1,
        104 / 103.5 - 1,
        104 / 103.5 - 1,
    ]
    np.testing.assert_allclose(result.values, expected, equal_nan=True)
    assert bool(result.support.iloc[0]) is False
    assert bool(result.support.iloc[-1]) is True
