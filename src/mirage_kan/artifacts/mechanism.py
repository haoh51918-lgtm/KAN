"""Deterministic mechanism cards and method-blind human review packages."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from mirage_kan.dsl import AstNode
from mirage_kan.dsl.core import LEAF_TYPES
from mirage_kan.mining.e3 import E3Atom, HardeningReceipt
from mirage_kan.mining.e3_runner import AtomPanel

MECHANISM_CARD_FIELDS = (
    "identity_and_canonical_ast",
    "raw_variables_windows_and_complexity",
    "edge_gate_and_shape_summary",
    "soft_hard_fidelity",
    "variable_and_lag_interventions",
    "local_counterfactual_response",
    "one_sentence_mechanism",
    "applicability_and_failure_conditions",
    "complete_kan_lineage_and_hardening_receipt",
)
REQUIRED_LINEAGE_FIELDS = (
    "miner",
    "global_seed",
    "checkpoint_sha256",
    "hardening_receipt_sha256",
)


def _ast_facts(ast: AstNode) -> dict[str, object]:
    variables: set[str] = set()
    windows: set[int] = set()

    def visit(node: AstNode) -> tuple[int, int, int]:
        if node.op in LEAF_TYPES:
            variables.add(node.op)
        if "window" in node.params:
            windows.add(int(node.params["window"]))
        child_facts = [visit(child) for child in node.children]
        nodes = 1 + sum(item[0] for item in child_facts)
        depth = 1 + max((item[1] for item in child_facts), default=0)
        cost = node.contract().cost_estimate + sum(item[2] for item in child_facts)
        return nodes, depth, cost

    nodes, depth, cost = visit(ast)
    contract = ast.validate()
    return {
        "raw_variables": sorted(variables),
        "windows": sorted(windows),
        "ast_nodes": nodes,
        "ast_depth": depth,
        "operator_cost": cost,
        "required_lookback": contract.lookback,
        "output_type": contract.output_type.value,
        "causal": contract.causal,
    }


def _finite_mapping(values: Mapping[str, object], name: str) -> dict[str, float]:
    if not values:
        raise ValueError(f"{name} must not be empty")
    result: dict[str, float] = {}
    for key, raw_value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"{name} values must be finite")
        result[key] = value
    return dict(sorted(result.items()))


def _choice_payload(choice: object, atom: E3Atom) -> dict[str, object]:
    return {
        "atom_index": choice.atom_index,
        "canonical_hash": choice.canonical_hash,
        "family": atom.family,
        "field": atom.field,
        "window": atom.window,
        "probability": choice.probability,
        "margin": choice.margin,
        "entropy": choice.entropy,
        "runner_up_atom_index": choice.runner_up_atom_index,
        "runner_up_canonical_hash": choice.runner_up_canonical_hash,
        "runner_up_probability": choice.runner_up_probability,
    }


def _primitive_shape(atom: E3Atom, edge_sign: int) -> dict[str, object]:
    ratio_grid = [round(0.5 + index * 0.01, 10) for index in range(101)]
    primitive = (
        [1.0 - value for value in ratio_grid]
        if atom.family == "lag_vs_price"
        else [value - 1.0 for value in ratio_grid]
    )
    contribution = [edge_sign * value for value in primitive]
    increasing = contribution[-1] > contribution[0]
    return {
        "primitive_input_ratio": ratio_grid,
        "primitive_output": primitive,
        "edge_contribution": contribution,
        "input_ratio_definition": {
            "return": "current_price_over_lagged_price",
            "price_vs_mean": "current_price_over_rolling_mean",
            "mean_vs_price": "rolling_mean_over_current_price",
            "lag_vs_price": "current_price_over_lagged_price",
            "volume_change": "current_volume_over_lagged_volume",
            "volume_vs_mean": "current_volume_over_rolling_mean",
        }[atom.family],
        "monotonic_interval": [0.5, 1.5],
        "monotonic_direction": "increasing" if increasing else "decreasing",
        "saturation_intervals": [],
        "shape_semantics": (
            "Exact analytic primitive response over a dimensionless input-ratio grid; "
            "the edge contribution includes the terminal Sub sign."
        ),
    }


def _family_hypothesis(atom: E3Atom) -> str:
    descriptions = {
        "return": "recent price continuation",
        "price_vs_mean": "price extension relative to its rolling mean",
        "mean_vs_price": "mean-reversion pressure from the current price gap",
        "lag_vs_price": "reversal pressure between lagged and current price",
        "volume_change": "change in trading activity and investor attention",
        "volume_vs_mean": "abnormal trading activity relative to its local norm",
    }
    return (
        f"{descriptions[atom.family]} measured on {atom.field} over "
        f"{atom.window} trading days"
    )


def _ablation_metrics(
    baseline: np.ndarray, intervened: np.ndarray
) -> tuple[float, float]:
    baseline_std = float(np.std(baseline, ddof=0))
    if not math.isfinite(baseline_std) or baseline_std <= 0.0:
        raise ValueError("baseline population standard deviation is not positive")
    normalized_delta = float(np.mean(np.abs(baseline - intervened))) / baseline_std
    intervened_std = float(np.std(intervened, ddof=0))
    if intervened_std == 0.0:
        correlation = 0.0
    elif np.array_equal(baseline, intervened):
        correlation = 1.0
    else:
        correlation = float(np.corrcoef(baseline, intervened)[0, 1])
        if not math.isfinite(correlation):
            raise ValueError("intervention Pearson correlation is not finite")
    return normalized_delta, correlation


def compute_mechanism_interventions(
    receipt: HardeningReceipt,
    atom_panel: AtomPanel,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Compute the frozen validation-only edge ablations and local responses."""
    if (
        atom_panel.dates.empty
        or atom_panel.dates.min().year != 2021
        or atom_panel.dates.max().year != 2021
    ):
        raise ValueError("mechanism interventions require the validation period")
    manifest = atom_panel.atom_manifest
    positive = manifest[receipt.positive.atom_index]
    negative = manifest[receipt.negative.atom_index]
    if (
        positive.canonical_hash != receipt.positive.canonical_hash
        or negative.canonical_hash != receipt.negative.canonical_hash
    ):
        raise ValueError("hardening receipt does not match the intervention atom panel")
    selected = (
        (positive, 1, receipt.positive.atom_index),
        (negative, -1, receipt.negative.atom_index),
    )
    valid = atom_panel.membership.copy()
    contributions: list[np.ndarray] = []
    for _, sign, atom_index in selected:
        values = atom_panel.values[..., atom_index]
        valid &= atom_panel.support[..., atom_index] & np.isfinite(values)
        contributions.append(float(sign) * values)
    if int(valid.sum()) < 2:
        raise ValueError("mechanism interventions have insufficient validation support")
    baseline = (contributions[0] + contributions[1])[valid]
    baseline_std = float(np.std(baseline, ddof=0))
    if not math.isfinite(baseline_std) or baseline_std <= 0.0:
        raise ValueError("baseline population standard deviation is not positive")

    def ablate(indices: set[int]) -> tuple[float, float]:
        intervened = sum(
            contribution if index not in indices else np.zeros_like(contribution)
            for index, contribution in enumerate(contributions)
        )[valid]
        return _ablation_metrics(baseline, intervened)

    variable_results: dict[str, float] = {}
    for field in sorted({positive.field, negative.field}):
        indices = {
            index for index, (atom, _, _) in enumerate(selected) if atom.field == field
        }
        normalized_delta, correlation = ablate(indices)
        variable_results[f"{field}.mean_absolute_delta_over_baseline_std"] = (
            normalized_delta
        )
        variable_results[f"{field}.baseline_intervened_pearson"] = correlation

    lag_results: dict[str, float] = {}
    for name, lower, upper in (
        ("short_2_5", 2, 5),
        ("medium_10_20", 10, 20),
        ("long_40_60", 40, 60),
    ):
        indices = {
            index
            for index, (atom, _, _) in enumerate(selected)
            if lower <= atom.window <= upper
        }
        normalized_delta, correlation = ablate(indices)
        lag_results[f"{name}.mean_absolute_delta_over_baseline_std"] = normalized_delta
        lag_results[f"{name}.baseline_intervened_pearson"] = correlation

    local_results: dict[str, float] = {}
    for edge_name, (_, sign, atom_index) in zip(
        ("positive", "negative"), selected, strict=True
    ):
        atom_values = atom_panel.values[..., atom_index][valid]
        atom_std = float(np.std(atom_values, ddof=0))
        if not math.isfinite(atom_std) or atom_std <= 0.0:
            raise ValueError(
                "selected atom population standard deviation is not positive"
            )
        for perturbation, direction in (("minus", -1.0), ("plus", 1.0)):
            delta = float(sign) * direction * atom_std
            prefix = f"{edge_name}_{perturbation}_one_sigma"
            local_results[f"{prefix}_mean_signed_factor_delta"] = delta
            local_results[f"{prefix}_response_sign"] = float(np.sign(delta))
    return variable_results, lag_results, local_results


def build_mechanism_card(
    *,
    factor_id: str,
    profile: str,
    receipt: HardeningReceipt,
    atom_manifest: Sequence[E3Atom],
    fidelity: Mapping[str, object],
    variable_interventions: Mapping[str, object],
    lag_interventions: Mapping[str, object],
    local_counterfactual_response: Mapping[str, object],
    lineage: Mapping[str, object],
) -> dict[str, object]:
    """Build one evidence-only card without making a human interpretability claim."""
    if not factor_id:
        raise ValueError("factor_id must not be empty")
    manifest = tuple(atom_manifest)
    if not manifest or {atom.profile for atom in manifest} != {profile}:
        raise ValueError("atom manifest does not match the declared profile")
    if [atom.atom_index for atom in manifest] != list(range(len(manifest))):
        raise ValueError("atom manifest indices are not canonical")
    positive = manifest[receipt.positive.atom_index]
    negative = manifest[receipt.negative.atom_index]
    if (
        positive.canonical_hash != receipt.positive.canonical_hash
        or negative.canonical_hash != receipt.negative.canonical_hash
        or receipt.ast.identity != AstNode("Sub", (positive.ast, negative.ast)).identity
    ):
        raise ValueError("hardening receipt does not match the atom manifest")

    fidelity_payload = _finite_mapping(fidelity, "soft-hard fidelity")
    if not {"pearson", "nrmse"}.issubset(fidelity_payload):
        raise ValueError("soft-hard fidelity requires pearson and nrmse")
    variables_payload = _finite_mapping(
        variable_interventions, "variable interventions"
    )
    lags_payload = _finite_mapping(lag_interventions, "lag interventions")
    counterfactual_payload = _finite_mapping(
        local_counterfactual_response, "local counterfactual response"
    )
    missing_lineage = set(REQUIRED_LINEAGE_FIELDS).difference(lineage)
    if missing_lineage:
        raise ValueError(f"complete KAN lineage is missing: {sorted(missing_lineage)}")

    ast_facts = _ast_facts(receipt.ast)
    positive_payload = _choice_payload(receipt.positive, positive)
    negative_payload = _choice_payload(receipt.negative, negative)
    mechanism = (
        f"The positive edge favors stronger {_family_hypothesis(positive)}, while the "
        f"negative edge subtracts {_family_hypothesis(negative)}; it may work when that cross-sectional "
        "imbalance captures persistent price pressure, reversal, or attention not "
        "already represented by the subtracted leg."
    )
    applicability = {
        "interpretation_level": "testable_mechanism_hypothesis_not_causal_claim",
        "applicability": (
            f"Requires causal observed {', '.join(ast_facts['raw_variables'])} history "
            f"with at least {ast_facts['required_lookback']} rows of lookback."
        ),
        "failure_conditions": (
            "Undefined denominators, incomplete required history, non-finite inputs, "
            "or an asset outside the point-in-time membership mask invalidate the score."
        ),
        "predevelopment_evidence_scope": (
            "Interventions and shape evidence use the frozen 2021 validation period; "
            "multi-year return, cost, and regime stability are assessed only after the "
            "single development opening and are not causal evidence."
        ),
    }
    hardening = {
        **dict(lineage),
        "profile": profile,
        "tau": receipt.tau,
        "checkpoint_logits_sha256": receipt.checkpoint_logits_sha256,
        "atom_manifest_sha256": receipt.atom_manifest_sha256,
        "selected_positive_atom_hash": positive.canonical_hash,
        "selected_negative_atom_hash": negative.canonical_hash,
        "rejected_alternates_count": len(receipt.rejected_alternates),
    }
    return {
        "identity_and_canonical_ast": {
            "factor_id": factor_id,
            "canonical_ast": receipt.ast.canonical_json(),
            "canonical_hash": receipt.ast.identity,
            "ast_identity": receipt.ast.identity,
        },
        "raw_variables_windows_and_complexity": ast_facts,
        "edge_gate_and_shape_summary": {
            "positive": positive_payload,
            "negative": negative_payload,
            "shape": "positive_categorical_edge_minus_negative_categorical_edge",
            "shape_plot_data": {
                "positive_edge": _primitive_shape(positive, 1),
                "negative_edge": _primitive_shape(negative, -1),
            },
        },
        "soft_hard_fidelity": fidelity_payload,
        "variable_and_lag_interventions": {
            "variable": variables_payload,
            "lag": lags_payload,
        },
        "local_counterfactual_response": counterfactual_payload,
        "one_sentence_mechanism": mechanism,
        "applicability_and_failure_conditions": applicability,
        "complete_kan_lineage_and_hardening_receipt": hardening,
    }


def build_blind_review_package(
    cards: Mapping[str, Mapping[str, object]],
    *,
    hides: Sequence[str],
    reviewers_minimum: int,
    response_direction_accuracy_minimum: float,
) -> dict[str, object]:
    """Create an allowlisted blind package that cannot expose method or PnL data."""
    if not cards:
        raise ValueError("blind package requires at least one mechanism card")
    if set(hides) != {"method_name", "pnl", "return_metrics"}:
        raise ValueError(
            "blind package hiding policy does not match the frozen protocol"
        )
    if type(reviewers_minimum) is not int or reviewers_minimum < 2:
        raise ValueError("blind review requires at least two reviewers")
    accuracy = float(response_direction_accuracy_minimum)
    if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
        raise ValueError("response direction accuracy must be finite and in [0, 1]")

    items: dict[str, dict[str, object]] = {}
    for index, factor_id in enumerate(sorted(cards), start=1):
        card = cards[factor_id]
        if set(card) != set(MECHANISM_CARD_FIELDS):
            raise ValueError("mechanism card fields do not match the frozen schema")
        identity = dict(card["identity_and_canonical_ast"])
        identity.pop("factor_id", None)
        edge = card["edge_gate_and_shape_summary"]
        if not isinstance(edge, Mapping):
            raise ValueError("mechanism edge summary must be a mapping")
        blinded_edge = {
            "positive_selected_atom": {
                key: edge["positive"][key]
                for key in ("family", "field", "window", "canonical_hash")
            },
            "negative_selected_atom": {
                key: edge["negative"][key]
                for key in ("family", "field", "window", "canonical_hash")
            },
            "shape": edge["shape"],
            "shape_plot_data": edge["shape_plot_data"],
        }
        positive = blinded_edge["positive_selected_atom"]
        negative = blinded_edge["negative_selected_atom"]
        items[f"B{index:03d}"] = {
            "identity_and_canonical_ast": identity,
            "raw_variables_windows_and_complexity": card[
                "raw_variables_windows_and_complexity"
            ],
            "edge_gate_and_shape_summary": blinded_edge,
            "variable_and_lag_interventions": card["variable_and_lag_interventions"],
            "response_direction_questions": [
                {
                    "question_id": "positive_edge_atom_plus_one_sigma",
                    "prompt": (
                        f"If the {positive['family']} atom output for "
                        f"{positive['field']} at window {positive['window']} increases "
                        "by one validation-population standard deviation while the "
                        "other edge is fixed, does the factor increase, decrease, "
                        "or remain unchanged?"
                    ),
                },
                {
                    "question_id": "negative_edge_atom_plus_one_sigma",
                    "prompt": (
                        f"If the {negative['family']} atom output for "
                        f"{negative['field']} at window {negative['window']} increases "
                        "by one validation-population standard deviation while the "
                        "other edge is fixed, does the factor increase, decrease, "
                        "or remain unchanged?"
                    ),
                },
            ],
            "one_sentence_mechanism": card["one_sentence_mechanism"],
            "applicability_and_failure_conditions": card[
                "applicability_and_failure_conditions"
            ],
        }
    return {
        "review_status": "pending_human_review",
        "reviewers_minimum": reviewers_minimum,
        "mechanism_restatement_required": True,
        "response_direction_accuracy_minimum": accuracy,
        "inter_reviewer_agreement_reported": True,
        "items": items,
    }


__all__ = [
    "MECHANISM_CARD_FIELDS",
    "build_blind_review_package",
    "build_mechanism_card",
    "compute_mechanism_interventions",
]
