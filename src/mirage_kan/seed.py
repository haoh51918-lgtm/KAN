"""Seed AST wiring-control library; these factors are not KAN-mined."""

from __future__ import annotations

from mirage_kan.dsl import AstNode


def seed_wiring_programs() -> dict[str, AstNode]:
    """Return a small deterministic library that exercises the complete S0 path."""
    leaf = AstNode
    return {
        "close_open": AstNode(
            "SafeDiv",
            (AstNode("Sub", (leaf("Close"), leaf("Open"))), leaf("Open")),
        ),
        "range_close": AstNode(
            "SafeDiv",
            (AstNode("Sub", (leaf("High"), leaf("Low"))), leaf("Close")),
        ),
        "return_5": AstNode("Return", (leaf("Close"),), {"window": 5}),
        "volume_mean_20": AstNode(
            "SafeDiv",
            (leaf("Volume"), AstNode("TsMean", (leaf("Volume"),), {"window": 20})),
        ),
    }
