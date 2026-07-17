"""Typed, causal AST contracts and deterministic Pandas execution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import pandas as pd

from mirage_kan.data.pit import PitPanel

WINDOWS = frozenset({2, 3, 5, 10, 20, 40, 60})
LEAF_TYPES = {
    "Open": "price_ts",
    "High": "price_ts",
    "Low": "price_ts",
    "Close": "price_ts",
    "Volume": "volume_ts",
}


class ProgramError(ValueError):
    """Raised when a program violates syntax, type, domain, or causality rules."""


class DslType(str, Enum):
    """Financial semantic and axis types used by the S0 DSL."""

    PRICE_TS = "price_ts"
    PRICE_DIFF_TS = "price_diff_ts"
    VOLUME_TS = "volume_ts"
    DIMENSIONLESS_TS = "dimensionless_ts"
    CROSS_SECTION_SIGNAL = "cross_section_signal"


@dataclass(frozen=True)
class OperatorContract:
    """Resolved contract for one complete AST node."""

    name: str
    arity: int
    input_types: tuple[DslType, ...]
    output_type: DslType
    lookback: int
    causal: bool
    commutative: bool
    domain: str
    mask_rule: str
    normalization_rule: str
    cost_estimate: int


@dataclass(frozen=True)
class Evaluation:
    """Expression output and expression support before universe/tradability filtering."""

    values: pd.Series
    support: pd.Series


def _window(params: Mapping[str, Any]) -> int:
    value = params.get("window")
    if type(value) is not int or value not in WINDOWS:
        raise ProgramError(f"window must be one of {sorted(WINDOWS)}, got {value!r}")
    return value


@dataclass(frozen=True)
class AstNode:
    """Immutable AST node with canonical identity derived from semantic content."""

    op: str
    children: tuple["AstNode", ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "children", tuple(self.children))
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))

    def validate(self) -> OperatorContract:
        """Resolve the node contract, recursively rejecting invalid programs."""
        child_contracts = tuple(child.validate() for child in self.children)
        child_types = tuple(contract.output_type for contract in child_contracts)
        child_lookback = max((contract.lookback for contract in child_contracts), default=1)

        if self.op in LEAF_TYPES:
            if self.children or self.params:
                raise ProgramError(f"leaf {self.op} cannot have children or parameters")
            return OperatorContract(
                self.op, 0, (), DslType(LEAF_TYPES[self.op]), 1, True, False,
                "raw value must be finite", "raw field observed mask",
                "none", 1,
            )
        if self.op == "Constant":
            value = self.params.get("value")
            if self.children or set(self.params) != {"value"}:
                raise ProgramError("Constant expects only the value parameter")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ProgramError("Constant value must be a finite real scalar")
            if not np.isfinite(value):
                raise ProgramError("Constant value must be a finite real scalar")
            return OperatorContract(
                self.op, 0, (), DslType.DIMENSIONLESS_TS, 1, True, False,
                "constant must be finite", "all panel rows", "none", 0,
            )
        if self.op not in {"Sub", "Add", "SafeDiv", "Delay", "Delta", "TsMean", "Return", "CSRank"}:
            raise ProgramError(f"unknown operator (future access is forbidden): {self.op}")

        arity = 2 if self.op in {"Sub", "Add", "SafeDiv"} else 1
        if len(self.children) != arity:
            raise ProgramError(f"{self.op} expects arity {arity}, got {len(self.children)}")
        if self.op in {"Delay", "Delta", "TsMean", "Return"}:
            window = _window(self.params)
        elif self.params:
            raise ProgramError(f"{self.op} does not accept parameters")

        if self.op in {"Sub", "Add"}:
            if child_types[0] is not child_types[1]:
                raise ProgramError(f"{self.op} type mismatch: {child_types}")
            if child_types[0] is DslType.CROSS_SECTION_SIGNAL:
                raise ProgramError(f"{self.op} type is not legal for cross-section signals")
            output = child_types[0]
            if self.op == "Sub" and output is DslType.PRICE_TS:
                output = DslType.PRICE_DIFF_TS
            return OperatorContract(
                self.op, 2, child_types, output, child_lookback, True,
                self.op == "Add", "children must be finite", "intersection of child support",
                "sort children" if self.op == "Add" else "ordered children", 1,
            )
        if self.op == "SafeDiv":
            legal = child_types[0] is child_types[1] or child_types == (
                DslType.PRICE_DIFF_TS, DslType.PRICE_TS
            )
            if not legal or DslType.CROSS_SECTION_SIGNAL in child_types:
                raise ProgramError(f"SafeDiv type mismatch: {child_types}")
            return OperatorContract(
                self.op, 2, child_types, DslType.DIMENSIONLESS_TS,
                child_lookback, True, False,
                "denominator must be finite and nonzero",
                "child support and nonzero denominator", "ordered children", 2,
            )
        child_type = child_types[0]
        if self.op == "Return":
            if child_type is not DslType.PRICE_TS:
                raise ProgramError(f"Return type must be price_ts, got {child_type}")
            return OperatorContract(
                self.op, 1, child_types, DslType.DIMENSIONLESS_TS,
                child_lookback + window, True, False,
                "lagged denominator must be finite and nonzero",
                "current and lagged child support", "fixed lag return", 2,
            )
        if self.op == "CSRank":
            if child_type is not DslType.DIMENSIONLESS_TS:
                raise ProgramError(f"CSRank type must be dimensionless_ts, got {child_type}")
            return OperatorContract(
                self.op, 1, child_types, DslType.CROSS_SECTION_SIGNAL,
                child_lookback, True, False, "at least one supported asset",
                "child support by date", "cross-sectional percentile rank", 2,
            )
        if child_type is DslType.CROSS_SECTION_SIGNAL:
            raise ProgramError(f"{self.op} cannot consume cross-section signal type")
        lookback = child_lookback + window if self.op in {"Delay", "Delta"} else child_lookback + window - 1
        output = DslType.PRICE_DIFF_TS if self.op == "Delta" and child_type is DslType.PRICE_TS else child_type
        mask = "current and lagged child support" if self.op in {"Delay", "Delta"} else "complete rolling child support"
        return OperatorContract(
            self.op, 1, child_types, output, lookback, True, False,
            "all required observations must be finite", mask, "fixed causal window", window,
        )

    def contract(self) -> OperatorContract:
        """Alias for validated contract resolution."""
        return self.validate()

    def _canonical_payload(self) -> dict[str, Any]:
        self.validate()
        children = [child._canonical_payload() for child in self.children]
        if self.op == "Add":
            children.sort(key=lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")))
        return {"op": self.op, "children": children, "params": dict(sorted(self.params.items()))}

    def canonical_json(self) -> str:
        """Return one whitespace-free canonical AST serialization."""
        return json.dumps(self._canonical_payload(), sort_keys=True, separators=(",", ":"))

    @property
    def identity(self) -> str:
        """Return the SHA-256 canonical AST identity."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible AST payload."""
        return self._canonical_payload()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AstNode":
        """Construct and validate an AST from a manifest payload."""
        if set(value) != {"op", "children", "params"}:
            raise ProgramError("AST payload must contain exactly op, children, and params")
        node = cls(
            str(value["op"]),
            tuple(cls.from_dict(child) for child in value["children"]),
            dict(value["params"]),
        )
        node.validate()
        return node


def _shift(series: pd.Series, periods: int) -> pd.Series:
    return series.groupby(level="instrument", sort=False).shift(periods)


def _rolling(series: pd.Series, window: int, operation: str) -> pd.Series:
    grouped = series.groupby(level="instrument", sort=False)
    rolled = grouped.rolling(window=window, min_periods=window)
    result = rolled.mean() if operation == "mean" else rolled.sum()
    return result.droplevel(0).reindex(series.index)


def evaluate(program: AstNode, panel: PitPanel) -> Evaluation:
    """Evaluate a validated AST deterministically from raw OHLCV only."""
    program.validate()

    def visit(node: AstNode) -> Evaluation:
        if node.op in LEAF_TYPES:
            return Evaluation(panel.field(node.op), panel.observed[node.op.lower()].copy())
        if node.op == "Constant":
            return Evaluation(
                pd.Series(float(node.params["value"]), index=panel.raw.index),
                pd.Series(True, index=panel.raw.index, name="support"),
            )
        children = tuple(visit(child) for child in node.children)
        if node.op in {"Sub", "Add"}:
            values = children[0].values - children[1].values if node.op == "Sub" else children[0].values + children[1].values
            support = children[0].support & children[1].support
        elif node.op == "SafeDiv":
            denominator = children[1].values
            domain = np.isfinite(denominator) & (denominator != 0)
            support = children[0].support & children[1].support & domain
            values = children[0].values / denominator.where(domain)
        elif node.op in {"Delay", "Delta", "Return"}:
            window = _window(node.params)
            lagged_values = _shift(children[0].values, window)
            lagged_support = _shift(children[0].support.astype(bool), window).fillna(False).astype(bool)
            support = children[0].support & lagged_support
            if node.op == "Delay":
                values = lagged_values
                support = lagged_support
            elif node.op == "Delta":
                values = children[0].values - lagged_values
            else:
                domain = np.isfinite(lagged_values) & (lagged_values != 0)
                support = support & domain
                values = children[0].values / lagged_values.where(domain) - 1.0
        elif node.op == "TsMean":
            window = _window(node.params)
            values = _rolling(children[0].values, window, "mean")
            count = _rolling(children[0].support.astype(int), window, "sum")
            support = count.eq(window)
        elif node.op == "CSRank":
            supported_values = children[0].values.where(children[0].support)
            values = supported_values.groupby(level="datetime", sort=False).rank(pct=True)
            support = children[0].support & values.notna()
        else:
            raise AssertionError(f"validated but unimplemented operator: {node.op}")
        finite = pd.Series(np.isfinite(values.to_numpy(dtype=float)), index=values.index)
        support = support.astype(bool) & finite
        return Evaluation(values.where(support), support.rename("support"))

    return visit(program)
