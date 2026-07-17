"""Pure-symbolic E3 categorical KAN edges over frozen typed atom banks."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Mapping, Sequence

import torch
from torch import nn

from mirage_kan.dsl import AstNode, DslType
from mirage_kan.experiments.gate_a.symbolic import fidelity_metrics

PRICE_FIELDS = ("Open", "High", "Low", "Close")
TRAINING_STEPS = 300
TEMPERATURE_INITIAL = 2.0
TEMPERATURE_FINAL = 0.10
HARD_ST_FINAL_FRACTION = 0.25
ENTROPY_COEFFICIENT = 0.001
EDGE_OVERLAP_COEFFICIENT = 0.01
FORWARD_MODES = ("soft", "hard_st")


@dataclass(frozen=True)
class ProfileSpec:
    """One frozen E3 profile definition."""

    windows: tuple[int, ...]
    fields: tuple[str, ...]
    atom_families: tuple[str, ...]


PROFILE_SPECS: Mapping[str, ProfileSpec] = MappingProxyType(
    {
        "short_price": ProfileSpec(
            (2, 3, 5, 10), PRICE_FIELDS, ("return", "price_vs_mean")
        ),
        "long_price": ProfileSpec(
            (10, 20, 40, 60), PRICE_FIELDS, ("return", "price_vs_mean")
        ),
        "reversal": ProfileSpec(
            (2, 3, 5, 10, 20),
            PRICE_FIELDS,
            ("mean_vs_price", "lag_vs_price"),
        ),
        "price_volume": ProfileSpec(
            (2, 3, 5, 10, 20, 40, 60),
            (*PRICE_FIELDS, "Volume"),
            ("return", "volume_change", "volume_vs_mean"),
        ),
    }
)


@dataclass(frozen=True)
class E3Atom:
    """A legal dimensionless temporal atom and its immutable identity."""

    atom_index: int
    profile: str
    family: str
    field: str
    window: int
    ast: AstNode
    canonical_hash: str


@dataclass(frozen=True)
class ChoiceReceipt:
    """Checkpoint-only diagnostics for one hardened categorical edge."""

    atom_index: int
    canonical_hash: str
    probability: float
    margin: float
    entropy: float
    runner_up_atom_index: int
    runner_up_canonical_hash: str
    runner_up_probability: float


@dataclass(frozen=True)
class RejectedAlternate:
    """One deterministic non-selected edge alternative."""

    edge: str
    atom_index: int
    canonical_hash: str
    probability: float
    reason: str


@dataclass(frozen=True)
class HardeningReceipt:
    """Auditable lowering of two checkpoint gates to one strict Sub AST."""

    ast: AstNode
    positive: ChoiceReceipt
    negative: ChoiceReceipt
    rejected_alternates: tuple[RejectedAlternate, ...]
    tau: float
    checkpoint_logits_sha256: str
    atom_manifest_sha256: str


def _leaf(field: str) -> AstNode:
    return AstNode(field)


def _centered_ratio(left: AstNode, right: AstNode) -> AstNode:
    return AstNode("SafeDiv", (AstNode("Sub", (left, right)), right))


def _atom_ast(family: str, field: str, window: int) -> AstNode:
    leaf = _leaf(field)
    mean = AstNode("TsMean", (leaf,), {"window": window})
    if family == "return" and field in PRICE_FIELDS:
        return AstNode("Return", (leaf,), {"window": window})
    if family == "price_vs_mean" and field in PRICE_FIELDS:
        return _centered_ratio(leaf, mean)
    if family == "mean_vs_price" and field in PRICE_FIELDS:
        return _centered_ratio(mean, leaf)
    if family == "lag_vs_price" and field in PRICE_FIELDS:
        lag = AstNode("Delay", (leaf,), {"window": window})
        return AstNode("SafeDiv", (AstNode("Sub", (lag, leaf)), lag))
    if family == "volume_change" and field == "Volume":
        delta = AstNode("Delta", (leaf,), {"window": window})
        lag = AstNode("Delay", (leaf,), {"window": window})
        return AstNode("SafeDiv", (delta, lag))
    if family == "volume_vs_mean" and field == "Volume":
        return _centered_ratio(leaf, mean)
    raise ValueError(f"atom family {family!r} is not legal for field {field!r}")


def _family_fields(family: str, spec: ProfileSpec) -> tuple[str, ...]:
    if family in {"return", "price_vs_mean", "mean_vs_price", "lag_vs_price"}:
        return tuple(field for field in spec.fields if field in PRICE_FIELDS)
    if family in {"volume_change", "volume_vs_mean"}:
        return ("Volume",) if "Volume" in spec.fields else ()
    raise ValueError(f"unknown frozen atom family: {family}")


@lru_cache(maxsize=None)
def build_profile_atom_bank(profile: str) -> tuple[E3Atom, ...]:
    """Construct one exact profile bank from the common typed DSL."""
    if profile not in PROFILE_SPECS:
        raise ValueError(f"unknown E3 profile: {profile!r}")
    spec = PROFILE_SPECS[profile]
    drafts: list[tuple[str, str, int, AstNode]] = []
    for family in spec.atom_families:
        for field in _family_fields(family, spec):
            for window in spec.windows:
                ast = _atom_ast(family, field, window)
                contract = ast.validate()
                if (
                    contract.output_type is not DslType.DIMENSIONLESS_TS
                    or not contract.causal
                ):
                    raise RuntimeError("frozen E3 atom is not causal and dimensionless")
                drafts.append((family, field, window, ast))
    deduplicated = {draft[3].identity: draft for draft in drafts}
    atoms = tuple(
        E3Atom(
            atom_index=atom_index,
            profile=profile,
            family=family,
            field=field,
            window=window,
            ast=ast,
            canonical_hash=canonical_hash,
        )
        for atom_index, (canonical_hash, (family, field, window, ast)) in enumerate(
            sorted(deduplicated.items())
        )
    )
    if not atoms:
        raise RuntimeError(f"profile {profile!r} produced an empty atom bank")
    return tuple(atoms)


def _validate_manifest(atom_manifest: Sequence[E3Atom]) -> tuple[E3Atom, ...]:
    manifest = tuple(atom_manifest)
    if not manifest:
        raise ValueError("atom manifest must not be empty")
    profiles = {atom.profile for atom in manifest}
    if len(profiles) != 1:
        raise ValueError("atom manifest must contain exactly one profile")
    if [atom.atom_index for atom in manifest] != list(range(len(manifest))):
        raise ValueError("atom manifest indices must be contiguous and ordered")
    for atom in manifest:
        contract = atom.ast.validate()
        if atom.canonical_hash != atom.ast.identity:
            raise ValueError("atom manifest canonical hash mismatch")
        if contract.output_type is not DslType.DIMENSIONLESS_TS or not contract.causal:
            raise ValueError("atom manifest contains an illegal typed atom")
    if len({atom.canonical_hash for atom in manifest}) != len(manifest):
        raise ValueError("atom manifest contains duplicate canonical identities")
    return manifest


def _validate_tau(tau: float) -> float:
    value = float(tau)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("tau must be finite and positive")
    return value


def _validate_logits(
    logits: torch.Tensor, atom_manifest: Sequence[E3Atom]
) -> tuple[E3Atom, ...]:
    manifest = _validate_manifest(atom_manifest)
    if tuple(logits.shape) != (2, len(manifest)):
        raise ValueError(f"logits must have shape (2, {len(manifest)})")
    if not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite floating-point values")
    return manifest


def _ordered_edge_indices(
    logits: torch.Tensor, edge: int, manifest: Sequence[E3Atom]
) -> tuple[int, ...]:
    values = logits[edge].detach().to(dtype=torch.float64, device="cpu").tolist()
    return tuple(
        sorted(
            range(len(manifest)),
            key=lambda index: (-values[index], manifest[index].canonical_hash),
        )
    )


def _select_distinct_indices(
    logits: torch.Tensor, manifest: Sequence[E3Atom]
) -> tuple[int, int]:
    values = logits.detach().to(dtype=torch.float64, device="cpu")
    pairs = (
        (positive, negative)
        for positive in range(len(manifest))
        for negative in range(len(manifest))
        if positive != negative
    )
    return min(
        pairs,
        key=lambda pair: (
            -float(values[0, pair[0]] + values[1, pair[1]]),
            manifest[pair[0]].canonical_hash,
            manifest[pair[1]].canonical_hash,
        ),
    )


def _straight_through_weights(
    probabilities: torch.Tensor, selected_index: int
) -> torch.Tensor:
    hard = torch.zeros_like(probabilities)
    hard[selected_index] = 1.0
    return hard - probabilities.detach() + probabilities


class CategoricalE3KAN(nn.Module):
    """Two differentiable categorical KAN edges over one frozen profile bank."""

    def __init__(
        self,
        profile: str,
        *,
        seed: int,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.atom_manifest = build_profile_atom_bank(profile)
        if dtype is not torch.float64:
            raise ValueError("the frozen E3 training dtype is torch.float64")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.gate_logits = nn.Parameter(
            torch.randn(
                2,
                len(self.atom_manifest),
                generator=generator,
                dtype=dtype,
            )
            * 0.01
        )

    @property
    def atom_count(self) -> int:
        return len(self.atom_manifest)

    def checkpoint_logits(self) -> torch.Tensor:
        """Return an immutable CPU copy suitable for checkpoint publication."""
        return self.gate_logits.detach().to(device="cpu").clone()

    def probabilities(self, tau: float) -> torch.Tensor:
        value = _validate_tau(tau)
        return torch.softmax(self.gate_logits / value, dim=-1)

    def edge_weights(self, tau: float, mode: str) -> torch.Tensor:
        if mode not in FORWARD_MODES:
            raise ValueError(f"mode must be one of {FORWARD_MODES}")
        probabilities = self.probabilities(tau)
        if mode == "soft":
            return probabilities
        positive, negative = _select_distinct_indices(
            self.gate_logits, self.atom_manifest
        )
        return torch.stack(
            (
                _straight_through_weights(probabilities[0], positive),
                _straight_through_weights(probabilities[1], negative),
            )
        )

    def forward(
        self, atom_values: torch.Tensor, *, tau: float, mode: str
    ) -> torch.Tensor:
        """Evaluate soft or hard-forward/soft-backward categorical edges."""
        if atom_values.ndim != 3 or atom_values.shape[-1] != self.atom_count:
            raise ValueError(
                f"atom values must have shape (dates, instruments, {self.atom_count})"
            )
        if not atom_values.is_floating_point() or not bool(
            torch.isfinite(atom_values).all()
        ):
            raise ValueError("atom values must be finite floating-point values")
        weights = self.edge_weights(tau, mode)
        return torch.matmul(atom_values, weights[0]) - torch.matmul(
            atom_values, weights[1]
        )


def temperature_at_step(update_index: int) -> float:
    """Return the fixed endpoint-inclusive 300-step linear temperature."""
    if type(update_index) is not int or not 0 <= update_index < TRAINING_STEPS:
        raise ValueError("update_index must be in [0, 300)")
    progress = update_index / (TRAINING_STEPS - 1)
    return TEMPERATURE_INITIAL + progress * (TEMPERATURE_FINAL - TEMPERATURE_INITIAL)


def forward_mode_at_step(update_index: int) -> str:
    """Use hard-ST for the final frozen quarter of optimizer updates."""
    temperature_at_step(update_index)
    hard_start = TRAINING_STEPS - int(TRAINING_STEPS * HARD_ST_FINAL_FRACTION)
    return "hard_st" if update_index >= hard_start else "soft"


def _mean_daily_cross_sectional_pearson(
    score: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if score.shape != target.shape or target.ndim != 2:
        raise ValueError("target shape must match the two-dimensional score shape")
    if valid_mask.shape != target.shape or valid_mask.dtype is not torch.bool:
        raise ValueError("valid mask shape must match target and use bool dtype")
    if not bool(torch.isfinite(score[valid_mask]).all()) or not bool(
        torch.isfinite(target[valid_mask]).all()
    ):
        raise ValueError("score and target must be finite on the valid mask")
    counts = valid_mask.sum(dim=1)
    safe_counts = counts.clamp_min(1).to(dtype=score.dtype)
    masked_score = torch.where(valid_mask, score, torch.zeros_like(score))
    masked_target = torch.where(valid_mask, target, torch.zeros_like(target))
    score_mean = masked_score.sum(dim=1) / safe_counts
    target_mean = masked_target.sum(dim=1) / safe_counts
    centered_score = torch.where(
        valid_mask, score - score_mean[:, None], torch.zeros_like(score)
    )
    centered_target = torch.where(
        valid_mask, target - target_mean[:, None], torch.zeros_like(target)
    )
    numerator = torch.sum(centered_score * centered_target, dim=1)
    denominator = torch.sqrt(
        torch.sum(centered_score.square(), dim=1)
        * torch.sum(centered_target.square(), dim=1)
    )
    usable = (counts >= 2) & (denominator > 0)
    if not bool(usable.any()):
        raise ValueError("no valid non-constant daily cross section is available")
    correlations = numerator[usable] / denominator[usable]
    return correlations.mean()


def training_objective(
    model: CategoricalE3KAN,
    atom_values: torch.Tensor,
    target: torch.Tensor,
    *,
    tau: float,
    mode: str,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute the frozen negative mean daily IC objective and penalties."""
    score = model(atom_values, tau=tau, mode=mode)
    if target.shape != score.shape:
        raise ValueError("target shape must match model score shape")
    mask = (
        torch.ones_like(target, dtype=torch.bool) if valid_mask is None else valid_mask
    )
    mean_daily_ic = _mean_daily_cross_sectional_pearson(score, target, mask)
    probabilities = model.probabilities(tau)
    entropy = -torch.sum(probabilities * torch.log(probabilities.clamp_min(1e-15)))
    edge_overlap = torch.sum(probabilities[0] * probabilities[1])
    total_loss = (
        -mean_daily_ic
        + ENTROPY_COEFFICIENT * entropy
        + EDGE_OVERLAP_COEFFICIENT * edge_overlap
    )
    return {
        "total_loss": total_loss,
        "mean_daily_ic": mean_daily_ic,
        "entropy": entropy,
        "edge_overlap": edge_overlap,
        "score": score,
    }


def atom_manifest_sha256(manifest: Sequence[E3Atom]) -> str:
    """Hash the complete ordered atom manifest used by a checkpoint."""
    manifest = _validate_manifest(manifest)
    payload = [
        {
            "atom_index": atom.atom_index,
            "profile": atom.profile,
            "family": atom.family,
            "field": atom.field,
            "window": atom.window,
            "ast": atom.ast.to_dict(),
            "canonical_hash": atom.canonical_hash,
        }
        for atom in manifest
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _logits_sha256(logits: torch.Tensor) -> str:
    values = logits.detach().to(dtype=torch.float64, device="cpu").contiguous()
    return hashlib.sha256(values.numpy().tobytes()).hexdigest()


def _choice_receipt(
    edge: int,
    selected: int,
    logits: torch.Tensor,
    probabilities: torch.Tensor,
    manifest: Sequence[E3Atom],
) -> ChoiceReceipt:
    order = _ordered_edge_indices(logits, edge, manifest)
    runner_up = next(index for index in order if index != selected)
    selected_probability = float(probabilities[edge, selected])
    runner_up_probability = float(probabilities[edge, runner_up])
    entropy = -torch.sum(
        probabilities[edge] * torch.log(probabilities[edge].clamp_min(1e-15))
    )
    return ChoiceReceipt(
        atom_index=selected,
        canonical_hash=manifest[selected].canonical_hash,
        probability=selected_probability,
        margin=selected_probability - runner_up_probability,
        entropy=float(entropy),
        runner_up_atom_index=runner_up,
        runner_up_canonical_hash=manifest[runner_up].canonical_hash,
        runner_up_probability=runner_up_probability,
    )


def harden_checkpoint(
    logits: torch.Tensor,
    atom_manifest: Sequence[E3Atom],
    tau: float,
) -> HardeningReceipt:
    """Lower checkpoint gates to one distinct-atom Sub AST without outcomes."""
    manifest = _validate_logits(logits, atom_manifest)
    value = _validate_tau(tau)
    detached = logits.detach().to(dtype=torch.float64, device="cpu")
    probabilities = torch.softmax(detached / value, dim=-1)
    positive, negative = _select_distinct_indices(detached, manifest)
    positive_order = _ordered_edge_indices(detached, 0, manifest)
    negative_order = _ordered_edge_indices(detached, 1, manifest)
    rejected: list[RejectedAlternate] = []
    if positive_order[0] == negative_order[0]:
        cancelled = positive_order[0]
        rejected.append(
            RejectedAlternate(
                edge="joint",
                atom_index=cancelled,
                canonical_hash=manifest[cancelled].canonical_hash,
                probability=float(
                    probabilities[0, cancelled] * probabilities[1, cancelled]
                ),
                reason="same_atom_cancellation",
            )
        )
    for edge_name, edge, selected, order in (
        ("positive", 0, positive, positive_order),
        ("negative", 1, negative, negative_order),
    ):
        alternate = next(index for index in order if index != selected)
        rejected.append(
            RejectedAlternate(
                edge=edge_name,
                atom_index=alternate,
                canonical_hash=manifest[alternate].canonical_hash,
                probability=float(probabilities[edge, alternate]),
                reason="lower_joint_checkpoint_rank",
            )
        )
    ast = AstNode("Sub", (manifest[positive].ast, manifest[negative].ast))
    contract = ast.validate()
    if (
        positive == negative
        or contract.output_type is not DslType.DIMENSIONLESS_TS
        or not contract.causal
    ):
        raise RuntimeError("hardener produced an illegal cancellation or output type")
    return HardeningReceipt(
        ast=ast,
        positive=_choice_receipt(0, positive, detached, probabilities, manifest),
        negative=_choice_receipt(1, negative, detached, probabilities, manifest),
        rejected_alternates=tuple(rejected),
        tau=value,
        checkpoint_logits_sha256=_logits_sha256(detached),
        atom_manifest_sha256=atom_manifest_sha256(manifest),
    )


def evaluate_hard_ast_from_atoms(
    ast: AstNode,
    atom_manifest: Sequence[E3Atom],
    atom_values: torch.Tensor,
) -> torch.Tensor:
    """Independently replay one hardened Sub AST on its exact atom tensor."""
    manifest = _validate_manifest(atom_manifest)
    if atom_values.ndim != 3 or atom_values.shape[-1] != len(manifest):
        raise ValueError(
            f"atom values must have shape (dates, instruments, {len(manifest)})"
        )
    if not bool(torch.isfinite(atom_values).all()):
        raise ValueError("atom values must be finite")
    ast.validate()
    if ast.op != "Sub" or len(ast.children) != 2:
        raise ValueError("hard AST must be exactly Sub(atom_positive, atom_negative)")
    by_hash = {atom.canonical_hash: atom.atom_index for atom in manifest}
    try:
        positive = by_hash[ast.children[0].identity]
        negative = by_hash[ast.children[1].identity]
    except KeyError as error:
        raise ValueError(
            "hard AST child is outside the frozen atom manifest"
        ) from error
    if positive == negative:
        raise ValueError("hard AST cannot cancel the same atom")
    return atom_values[..., positive] - atom_values[..., negative]


def soft_hard_fidelity(
    soft_predictions: torch.Tensor, hard_predictions: torch.Tensor
) -> dict[str, float]:
    """Return the frozen soft-hard Pearson, NRMSE, and maximum error metrics."""
    return fidelity_metrics(soft_predictions, hard_predictions)


__all__ = [
    "CategoricalE3KAN",
    "ChoiceReceipt",
    "E3Atom",
    "HardeningReceipt",
    "PROFILE_SPECS",
    "RejectedAlternate",
    "atom_manifest_sha256",
    "build_profile_atom_bank",
    "evaluate_hard_ast_from_atoms",
    "forward_mode_at_step",
    "harden_checkpoint",
    "soft_hard_fidelity",
    "temperature_at_step",
    "training_objective",
]
