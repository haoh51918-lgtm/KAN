"""Capacity-matched black-box falsification control for S2a v2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from mirage_kan.data import PitPanel
from mirage_kan.mining.e3 import (
    PROFILE_SPECS,
    atom_manifest_sha256,
    build_profile_atom_bank,
)
from mirage_kan.mining.e3_runner import (
    BOOTSTRAP_SEED_BASE,
    LEARNING_RATE,
    MINERS_PER_PROFILE,
    TRAINING_STEPS,
    AtomPanel,
    BlockBootstrapReceipt,
    draw_training_bootstrap,
    materialize_atom_panel,
)

HIDDEN_WIDTH = 2
MLP_SEED_BASE = 32452843
PARAMETER_GAP_MAXIMUM = 0.10


def _validate_pair_identity(profile: str, global_attempt_index: int) -> None:
    if profile not in PROFILE_SPECS:
        raise ValueError(f"unknown E3 profile: {profile!r}")
    if type(global_attempt_index) is not int or not (
        0 <= global_attempt_index < len(PROFILE_SPECS) * MINERS_PER_PROFILE
    ):
        raise ValueError("KAN global attempt index must be in [0, 256)")
    expected_profile = tuple(PROFILE_SPECS)[global_attempt_index // MINERS_PER_PROFILE]
    if profile != expected_profile:
        raise ValueError(
            "profile does not match the frozen KAN global-attempt profile segment"
        )


@dataclass(frozen=True)
class MLPControlPairing:
    """Immutable link from one selected KAN factor to its exact control budget."""

    profile: str
    kan_global_attempt_index: int
    bootstrap: BlockBootstrapReceipt

    def __post_init__(self) -> None:
        _validate_pair_identity(self.profile, self.kan_global_attempt_index)
        if not isinstance(self.bootstrap, BlockBootstrapReceipt):
            raise TypeError("bootstrap must be a BlockBootstrapReceipt")

    @staticmethod
    def validate_many(
        pairings: Sequence[MLPControlPairing],
        *,
        minimum_library_size: int,
        library_cap: int,
    ) -> tuple[MLPControlPairing, ...]:
        """Freeze a production pairing set and reject post-hoc duplication."""
        frozen = tuple(pairings)
        indices = [item.kan_global_attempt_index for item in frozen]
        if len(set(indices)) != len(indices):
            raise ValueError("paired KAN global attempt indices must be unique")
        if (
            type(minimum_library_size) is not int
            or type(library_cap) is not int
            or not 1 <= minimum_library_size <= len(frozen) <= library_cap
        ):
            raise ValueError(
                "production pairing count violates frozen admission bounds"
            )
        return frozen


class MatchedBlackboxMLP(nn.Module):
    """Frozen one-hidden-layer, two-unit SiLU capacity control."""

    def __init__(self, profile: str, kan_global_attempt_index: int) -> None:
        super().__init__()
        _validate_pair_identity(profile, kan_global_attempt_index)
        self.profile = profile
        self.kan_global_attempt_index = kan_global_attempt_index
        self.seed = MLP_SEED_BASE + kan_global_attempt_index
        self.hidden_width = HIDDEN_WIDTH
        self.atom_count = len(build_profile_atom_bank(profile))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.network = nn.Sequential(
                nn.Linear(self.atom_count, HIDDEN_WIDTH),
                nn.SiLU(),
                nn.Linear(HIDDEN_WIDTH, 1),
            )
        self.double()
        if self.parameter_relative_gap > PARAMETER_GAP_MAXIMUM:
            raise ValueError(
                "two-unit MLP parameter count is not within 10% of its paired KAN"
            )

    @property
    def kan_parameter_count(self) -> int:
        """The paired categorical KAN has two logits per profile atom."""
        return 2 * self.atom_count

    @property
    def parameter_count(self) -> int:
        """Count all and only trainable black-box parameters."""
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    @property
    def parameter_relative_gap(self) -> float:
        """Absolute capacity gap relative to the paired KAN."""
        return abs(self.parameter_count - self.kan_parameter_count) / float(
            self.kan_parameter_count
        )

    def forward(self, atom_values: torch.Tensor) -> torch.Tensor:
        """Score a date-major tensor of the paired profile's atoms."""
        if atom_values.ndim != 3 or atom_values.shape[-1] != self.atom_count:
            raise ValueError(
                f"atom values must have shape (dates, instruments, {self.atom_count})"
            )
        if atom_values.dtype is not torch.float64 or not bool(
            torch.isfinite(atom_values).all()
        ):
            raise ValueError("atom values must be finite float64")
        return self.network(atom_values).squeeze(-1)


@dataclass(frozen=True)
class MLPTrainingStepReceipt:
    """One optimizer update and the complete resulting MLP checkpoint."""

    update_index: int
    total_loss: float
    mean_daily_ic: float
    parameters: torch.Tensor


@dataclass(frozen=True)
class MLPControlReceipt:
    """Auditable fitted output for one KAN-paired black-box control."""

    profile: str
    kan_global_attempt_index: int
    seed: int
    bootstrap: BlockBootstrapReceipt
    optimizer: str
    learning_rate: float
    scheduled_updates: int
    completed_updates: int
    input_atom_count: int
    kan_parameter_count: int
    mlp_parameter_count: int
    parameter_relative_gap: float
    atom_manifest_sha256: str
    valid_support_sha256: str
    initial_parameters: torch.Tensor
    final_parameters: torch.Tensor
    first_step_data_gradient: torch.Tensor
    trajectory: tuple[MLPTrainingStepReceipt, ...]
    prediction: torch.Tensor
    prediction_mask: torch.Tensor
    kan_mined: bool = field(default=False, init=False)
    promotion_eligible: bool = field(default=False, init=False)
    factor_library_publication_allowed: bool = field(default=False, init=False)


@dataclass(frozen=True)
class MatchedBlackboxControlPanel:
    """Control-only output container that cannot masquerade as a factor library."""

    controls: tuple[MLPControlReceipt, ...]
    role: str = field(default="falsification_control_never_production", init=False)
    output_kind: str = field(default="control_panel_not_factor_library", init=False)
    promotion_eligible: bool = field(default=False, init=False)
    factor_library_publication_allowed: bool = field(default=False, init=False)


def _parameter_vector(model: nn.Module) -> torch.Tensor:
    return (
        torch.cat(
            tuple(
                parameter.detach().reshape(-1)
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        )
        .to(device="cpu", dtype=torch.float64)
        .clone()
    )


def _gradient_vector(gradients: Sequence[torch.Tensor]) -> torch.Tensor:
    return (
        torch.cat(tuple(item.detach().reshape(-1) for item in gradients))
        .to(device="cpu", dtype=torch.float64)
        .clone()
    )


def _weighted_mean_daily_ic(
    score: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    multiplicity_weights: torch.Tensor,
) -> torch.Tensor:
    """Mean daily cross-sectional Pearson IC with exact date multiplicities."""
    if score.shape != target.shape or target.ndim != 2:
        raise ValueError("score and target must share a two-dimensional shape")
    if valid_mask.shape != target.shape or valid_mask.dtype is not torch.bool:
        raise ValueError("valid mask must be bool with the target shape")
    if multiplicity_weights.shape != (target.shape[0],):
        raise ValueError("multiplicity weights must have one value per date")
    if not (
        score.device
        == target.device
        == valid_mask.device
        == multiplicity_weights.device
    ):
        raise ValueError("all objective tensors must use the same device")
    if not bool(torch.isfinite(score[valid_mask]).all()) or not bool(
        torch.isfinite(target[valid_mask]).all()
    ):
        raise ValueError("score and target must be finite on the valid support")
    if not bool(torch.isfinite(multiplicity_weights).all()) or bool(
        (multiplicity_weights < 0).any()
    ):
        raise ValueError("multiplicity weights must be finite and nonnegative")

    counts = valid_mask.sum(dim=1)
    safe_counts = counts.clamp_min(1).to(dtype=torch.float64)
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
    effective_weights = multiplicity_weights * usable.to(dtype=torch.float64)
    weight_sum = effective_weights.sum()
    if not bool(torch.isfinite(weight_sum)) or float(weight_sum) <= 0.0:
        raise ValueError("paired bootstrap has no usable daily cross section")
    daily_ic = torch.where(usable, numerator / denominator.clamp_min(1e-300), 0.0)
    return torch.sum(daily_ic * effective_weights) / weight_sum


def _tensor_sha256(tensor: torch.Tensor) -> str:
    values = tensor.detach().to(device="cpu").contiguous()
    header = json.dumps(
        {"shape": list(values.shape), "dtype": str(values.dtype)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(header + values.numpy().tobytes()).hexdigest()


def _validate_training_inputs(
    model: MatchedBlackboxMLP,
    atom_values: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None:
    expected_atom_shape = (*target.shape, model.atom_count)
    if target.ndim != 2 or tuple(atom_values.shape) != expected_atom_shape:
        raise ValueError(
            f"atom values must have shape {expected_atom_shape} for target shape"
        )
    if atom_values.dtype is not torch.float64 or not bool(
        torch.isfinite(atom_values).all()
    ):
        raise ValueError("atom values must be finite float64")
    if valid_mask.shape != target.shape or valid_mask.dtype is not torch.bool:
        raise ValueError("valid mask must be bool with the target shape")
    if atom_values.device != target.device or target.device != valid_mask.device:
        raise ValueError("input tensors must use the same device")
    if target.dtype is not torch.float64 or not bool(
        torch.isfinite(target[valid_mask]).all()
    ):
        raise ValueError("target must be float64 and finite on valid support")


def _validate_exact_bootstrap(
    receipt: BlockBootstrapReceipt,
    *,
    date_count: int,
    kan_global_attempt_index: int,
) -> None:
    expected = draw_training_bootstrap(
        date_count, BOOTSTRAP_SEED_BASE + kan_global_attempt_index
    )
    if receipt != expected:
        raise ValueError("control did not receive the exact paired KAN bootstrap")


def _train_control(
    *,
    profile: str,
    kan_global_attempt_index: int,
    bootstrap: BlockBootstrapReceipt,
    atom_values: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    prediction_mask: torch.Tensor,
    steps: int,
    device: str | torch.device,
) -> MLPControlReceipt:
    if type(steps) is not int or not 0 <= steps <= TRAINING_STEPS:
        raise ValueError("steps must not exceed the frozen 300-update scheduled budget")
    model = MatchedBlackboxMLP(profile, kan_global_attempt_index)
    _validate_training_inputs(model, atom_values, target, valid_mask)
    if prediction_mask.shape != target.shape or prediction_mask.dtype is not torch.bool:
        raise ValueError("prediction mask must be bool with the target shape")
    _validate_exact_bootstrap(
        bootstrap,
        date_count=target.shape[0],
        kan_global_attempt_index=kan_global_attempt_index,
    )

    target_device = torch.device(device)
    model = model.to(device=target_device)
    atoms = atom_values.detach().to(device=target_device)
    outcomes = target.detach().to(device=target_device)
    mask = valid_mask.detach().to(device=target_device)
    weights = torch.tensor(
        bootstrap.multiplicities, dtype=torch.float64, device=target_device
    )
    initial_parameters = _parameter_vector(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    first_data_gradient: torch.Tensor | None = None
    trajectory: list[MLPTrainingStepReceipt] = []
    for update_index in range(steps):
        optimizer.zero_grad(set_to_none=True)
        score = model(atoms)
        mean_daily_ic = _weighted_mean_daily_ic(score, outcomes, mask, weights)
        loss = -mean_daily_ic
        parameters = tuple(
            parameter for parameter in model.parameters() if parameter.requires_grad
        )
        if update_index == 0:
            first_data_gradient = _gradient_vector(
                torch.autograd.grad(loss, parameters, retain_graph=True)
            )
        loss.backward()
        if any(
            parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
            for parameter in parameters
        ):
            raise FloatingPointError("MLP data gradient is missing or non-finite")
        optimizer.step()
        parameter_vector = _parameter_vector(model)
        if not bool(torch.isfinite(parameter_vector).all()):
            raise FloatingPointError("MLP update produced non-finite parameters")
        trajectory.append(
            MLPTrainingStepReceipt(
                update_index=update_index,
                total_loss=float(loss.detach().to(device="cpu")),
                mean_daily_ic=float(mean_daily_ic.detach().to(device="cpu")),
                parameters=parameter_vector,
            )
        )

    final_parameters = _parameter_vector(model)
    if first_data_gradient is None:
        first_data_gradient = torch.zeros_like(final_parameters)
    with torch.no_grad():
        prediction = model(atoms).detach().to(device="cpu").clone()
    return MLPControlReceipt(
        profile=profile,
        kan_global_attempt_index=kan_global_attempt_index,
        seed=model.seed,
        bootstrap=bootstrap,
        optimizer="Adam",
        learning_rate=LEARNING_RATE,
        scheduled_updates=TRAINING_STEPS,
        completed_updates=steps,
        input_atom_count=model.atom_count,
        kan_parameter_count=model.kan_parameter_count,
        mlp_parameter_count=model.parameter_count,
        parameter_relative_gap=model.parameter_relative_gap,
        atom_manifest_sha256=atom_manifest_sha256(build_profile_atom_bank(profile)),
        valid_support_sha256=_tensor_sha256(valid_mask),
        initial_parameters=initial_parameters,
        final_parameters=final_parameters,
        first_step_data_gradient=first_data_gradient,
        trajectory=tuple(trajectory),
        prediction=prediction,
        prediction_mask=prediction_mask.detach().to(device="cpu").clone(),
    )


def _control_tensors(
    atom_panel: AtomPanel, target: pd.Series
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(target.index, pd.MultiIndex) or target.index.has_duplicates:
        raise ValueError("target must have a unique MultiIndex")
    aligned_target = (
        target.reindex(atom_panel.index)
        .to_numpy(dtype=np.float64, copy=True)
        .reshape(len(atom_panel.dates), len(atom_panel.instruments))
    )
    target_support = np.isfinite(aligned_target)
    valid_mask = atom_panel.joint_support & atom_panel.membership & target_support
    prediction_mask = atom_panel.joint_support & atom_panel.membership
    if not np.any(valid_mask):
        raise ValueError("joint atom, membership, and target support is empty")
    objective_dates = np.any(valid_mask, axis=1)
    aligned_target = aligned_target[objective_dates]
    target_support = target_support[objective_dates]
    valid_mask = valid_mask[objective_dates]
    prediction_mask = prediction_mask[objective_dates]
    finite_atom_values = np.where(
        atom_panel.support[objective_dates],
        atom_panel.values[objective_dates],
        0.0,
    )
    finite_target = np.where(target_support, aligned_target, 0.0)
    return (
        torch.from_numpy(np.array(finite_atom_values, dtype=np.float64, copy=True)),
        torch.from_numpy(np.array(finite_target, dtype=np.float64, copy=True)),
        torch.from_numpy(np.array(valid_mask, dtype=bool, copy=True)),
        torch.from_numpy(np.array(prediction_mask, dtype=bool, copy=True)),
    )


def run_matched_blackbox_controls(
    panel: PitPanel,
    target: pd.Series,
    pairings: Sequence[MLPControlPairing],
    *,
    minimum_library_size: int,
    library_cap: int,
    device: str | torch.device = "cpu",
) -> MatchedBlackboxControlPanel:
    """Fit exactly one frozen-budget MLP for each selected KAN factor."""
    maximum_date = pd.to_datetime(panel.raw.index.get_level_values("datetime")).max()
    if maximum_date > pd.Timestamp("2020-12-31"):
        raise ValueError("MLP control training panel crosses the frozen train boundary")
    frozen_pairings = MLPControlPairing.validate_many(
        pairings,
        minimum_library_size=minimum_library_size,
        library_cap=library_cap,
    )
    profile_tensors: dict[
        str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ] = {}
    controls: list[MLPControlReceipt] = []
    for pairing in frozen_pairings:
        if pairing.profile not in profile_tensors:
            atom_panel = materialize_atom_panel(panel, pairing.profile)
            profile_tensors[pairing.profile] = _control_tensors(atom_panel, target)
        atoms, outcomes, valid_mask, prediction_mask = profile_tensors[pairing.profile]
        controls.append(
            _train_control(
                profile=pairing.profile,
                kan_global_attempt_index=pairing.kan_global_attempt_index,
                bootstrap=pairing.bootstrap,
                atom_values=atoms,
                target=outcomes,
                valid_mask=valid_mask,
                prediction_mask=prediction_mask,
                steps=TRAINING_STEPS,
                device=device,
            )
        )
    return MatchedBlackboxControlPanel(tuple(controls))


def replay_control_on_atom_panel(
    control: MLPControlReceipt, atom_panel: AtomPanel
) -> pd.Series:
    """Independently replay one completed control on any same-profile atom panel."""
    if (
        control.completed_updates != TRAINING_STEPS
        or len(control.trajectory) != TRAINING_STEPS
        or [step.update_index for step in control.trajectory]
        != list(range(TRAINING_STEPS))
    ):
        raise ValueError("control replay requires a complete 300-update receipt")
    if control.profile != atom_panel.profile:
        raise ValueError("control and atom panel profiles differ")
    if control.seed != MLP_SEED_BASE + control.kan_global_attempt_index:
        raise ValueError("control seed does not match its paired KAN attempt")
    if control.atom_manifest_sha256 != atom_manifest_sha256(atom_panel.atom_manifest):
        raise ValueError("control atom manifest differs from the replay panel")
    model = MatchedBlackboxMLP(control.profile, control.kan_global_attempt_index)
    if control.final_parameters.shape != (model.parameter_count,) or not bool(
        torch.isfinite(control.final_parameters).all()
    ):
        raise ValueError("control final parameter vector is invalid")
    offset = 0
    with torch.no_grad():
        for parameter in model.parameters():
            count = parameter.numel()
            parameter.copy_(
                control.final_parameters[offset : offset + count].reshape(
                    parameter.shape
                )
            )
            offset += count
        values = np.where(atom_panel.support, atom_panel.values, 0.0)
        prediction = model(
            torch.from_numpy(np.array(values, dtype=np.float64, copy=True))
        ).numpy()
    support = atom_panel.joint_support & atom_panel.membership
    return pd.Series(
        prediction.reshape(-1),
        index=atom_panel.index,
        name=f"mlp_for_kan_{control.kan_global_attempt_index:03d}",
    ).where(support.reshape(-1))


def _run_tiny_matched_control_for_test(
    *,
    profile: str,
    kan_global_attempt_index: int,
    bootstrap: BlockBootstrapReceipt,
    atom_values: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    steps: int,
    device: str | torch.device,
) -> MLPControlReceipt:
    """Exercise control mechanics at reduced scale; never used by production."""
    return _train_control(
        profile=profile,
        kan_global_attempt_index=kan_global_attempt_index,
        bootstrap=bootstrap,
        atom_values=atom_values,
        target=target,
        valid_mask=valid_mask,
        prediction_mask=valid_mask,
        steps=steps,
        device=device,
    )


__all__ = [
    "MLPControlPairing",
    "MLPControlReceipt",
    "MLPTrainingStepReceipt",
    "MatchedBlackboxControlPanel",
    "MatchedBlackboxMLP",
    "replay_control_on_atom_panel",
    "run_matched_blackbox_controls",
]
