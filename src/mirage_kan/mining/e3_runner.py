"""Batched, replayable training runner for the frozen E3 categorical KAN miner."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from mirage_kan.data import PitPanel
from mirage_kan.dsl import AstNode, evaluate
from mirage_kan.mining.e3 import (
    CategoricalE3KAN,
    E3Atom,
    HardeningReceipt,
    PROFILE_SPECS,
    build_profile_atom_bank,
    evaluate_hard_ast_from_atoms,
    forward_mode_at_step,
    harden_checkpoint,
    soft_hard_fidelity,
    temperature_at_step,
)

MINERS_PER_PROFILE = 64
TRAINING_STEPS = 300
LEARNING_RATE = 0.03
MINER_SEED_BASE = 730001
BOOTSTRAP_SEED_BASE = 49979687
BOOTSTRAP_FRACTION = 0.80
BLOCK_LENGTH = 20
FINAL_TEMPERATURE = 0.10
ENTROPY_COEFFICIENT = 0.001
EDGE_OVERLAP_COEFFICIENT = 0.01
FIDELITY_PEARSON_MINIMUM = 0.98
FIDELITY_NRMSE_MAXIMUM = 0.10
GATE_MARGIN_MINIMUM = 0.05


@dataclass(frozen=True)
class AtomPanel:
    """Date-major rectangular values and exact support for one frozen atom bank."""

    profile: str
    atom_manifest: tuple[E3Atom, ...]
    dates: pd.DatetimeIndex
    instruments: pd.Index
    index: pd.MultiIndex
    values: np.ndarray
    support: np.ndarray
    joint_support: np.ndarray
    membership: np.ndarray


@dataclass(frozen=True)
class BlockBootstrapReceipt:
    """Complete replay record for one non-wrapping moving-block draw."""

    seed: int
    rng: str
    date_count: int
    fraction: float
    block_length: int
    target_date_draws: int
    drawn_block_starts: tuple[int, ...]
    sampled_date_indices: tuple[int, ...]
    multiplicities: tuple[int, ...]


@dataclass(frozen=True)
class TrainingStepReceipt:
    """One optimizer update and the resulting complete gate checkpoint."""

    update_index: int
    tau: float
    mode: str
    total_loss: float
    mean_daily_ic: float
    entropy: float
    edge_overlap: float
    gate_logits: torch.Tensor


@dataclass(frozen=True)
class MinerReceipt:
    """One and only one candidate plus its complete KAN training lineage."""

    profile: str
    profile_index: int
    miner_index: int
    global_attempt_index: int
    miner_seed: int
    bootstrap: BlockBootstrapReceipt
    initial_logits: torch.Tensor
    final_logits: torch.Tensor
    first_step_data_gradient: torch.Tensor
    trajectory: tuple[TrainingStepReceipt, ...]
    hardening: HardeningReceipt
    candidate_ast: AstNode
    fidelity: dict[str, float]
    admission_failures: tuple[str, ...]


@dataclass(frozen=True)
class ProfileRun:
    """The fixed population result for one E3 profile."""

    profile: str
    device: str
    miners: tuple[MinerReceipt, ...]


@dataclass(frozen=True)
class _BatchStep:
    update_index: int
    tau: float
    mode: str
    total_loss: torch.Tensor
    mean_daily_ic: torch.Tensor
    entropy: torch.Tensor
    edge_overlap: torch.Tensor
    gate_logits: torch.Tensor


def _profile_index(profile: str) -> int:
    try:
        return tuple(PROFILE_SPECS).index(profile)
    except ValueError as error:
        raise ValueError(f"unknown E3 profile: {profile!r}") from error


def materialize_atom_panel(panel: PitPanel, profile: str) -> AtomPanel:
    """Evaluate every atom independently with the public DSL and retain missingness."""
    manifest = build_profile_atom_bank(profile)
    dates = pd.DatetimeIndex(
        panel.raw.index.get_level_values("datetime").unique()
    ).sort_values()
    instruments = pd.Index(
        panel.raw.index.get_level_values("instrument").unique(), name="instrument"
    ).sort_values()
    index = pd.MultiIndex.from_product(
        (dates, instruments), names=("datetime", "instrument")
    )
    flat_values: list[np.ndarray] = []
    flat_support: list[np.ndarray] = []
    for atom in manifest:
        result = evaluate(atom.ast, panel)
        support = result.support.reindex(index, fill_value=False).to_numpy(
            dtype=bool, copy=True
        )
        values = result.values.reindex(index).to_numpy(dtype=np.float64, copy=True)
        values[~support] = np.nan
        flat_values.append(values)
        flat_support.append(support)

    date_count = len(dates)
    instrument_count = len(instruments)
    values = np.stack(flat_values, axis=-1).reshape(
        date_count, instrument_count, len(manifest)
    )
    support = np.stack(flat_support, axis=-1).reshape(values.shape)
    joint_support = support.all(axis=-1)
    membership = (
        panel.membership.reindex(index)
        .astype("boolean")
        .fillna(False)
        .to_numpy(dtype=bool, copy=True)
        .reshape(date_count, instrument_count)
    )
    values.setflags(write=False)
    support.setflags(write=False)
    joint_support.setflags(write=False)
    membership.setflags(write=False)
    return AtomPanel(
        profile=profile,
        atom_manifest=manifest,
        dates=dates,
        instruments=instruments,
        index=index,
        values=values,
        support=support,
        joint_support=joint_support,
        membership=membership,
    )


def _draw_blocks(
    *, date_count: int, seed: int, target_date_draws: int, block_length: int
) -> BlockBootstrapReceipt:
    if type(date_count) is not int or date_count < 1:
        raise ValueError("date_count must be a positive integer")
    if type(block_length) is not int or not 1 <= block_length <= date_count:
        raise ValueError("block_length must be in [1, date_count]")
    if type(target_date_draws) is not int or target_date_draws < 1:
        raise ValueError("target_date_draws must be a positive integer")
    block_count = math.ceil(target_date_draws / block_length)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    starts = rng.integers(
        0,
        date_count - block_length + 1,
        size=block_count,
        dtype=np.int64,
    )
    sampled = np.concatenate(
        tuple(
            np.arange(start, start + block_length, dtype=np.int64) for start in starts
        )
    )[:target_date_draws]
    multiplicities = np.bincount(sampled, minlength=date_count)
    return BlockBootstrapReceipt(
        seed=int(seed),
        rng="numpy.random.Generator(PCG64)",
        date_count=date_count,
        fraction=BOOTSTRAP_FRACTION,
        block_length=block_length,
        target_date_draws=target_date_draws,
        drawn_block_starts=tuple(int(value) for value in starts),
        sampled_date_indices=tuple(int(value) for value in sampled),
        multiplicities=tuple(int(value) for value in multiplicities),
    )


def draw_training_bootstrap(date_count: int, seed: int) -> BlockBootstrapReceipt:
    """Draw the frozen ceil(0.8D), 20-day, PCG64 non-wrapping bootstrap."""
    return _draw_blocks(
        date_count=date_count,
        seed=seed,
        target_date_draws=math.ceil(BOOTSTRAP_FRACTION * date_count),
        block_length=BLOCK_LENGTH,
    )


def _validate_batch_inputs(
    logits: torch.Tensor,
    atom_manifest: Sequence[E3Atom],
    atom_values: torch.Tensor,
) -> tuple[E3Atom, ...]:
    manifest = tuple(atom_manifest)
    if not manifest or [atom.atom_index for atom in manifest] != list(
        range(len(manifest))
    ):
        raise ValueError("atom manifest must be nonempty, contiguous, and ordered")
    hashes = [atom.canonical_hash for atom in manifest]
    if hashes != sorted(hashes) or len(set(hashes)) != len(hashes):
        raise ValueError("atom manifest must use unique canonical-hash order")
    if logits.ndim != 3 or tuple(logits.shape[1:]) != (2, len(manifest)):
        raise ValueError(f"logits must have shape (miners, 2, {len(manifest)})")
    if logits.dtype is not torch.float64 or not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite float64")
    if atom_values.ndim != 3 or atom_values.shape[-1] != len(manifest):
        raise ValueError(
            f"atom values must have shape (dates, instruments, {len(manifest)})"
        )
    if atom_values.dtype is not torch.float64 or not bool(
        torch.isfinite(atom_values).all()
    ):
        raise ValueError("atom values must be finite float64")
    if logits.device != atom_values.device:
        raise ValueError("logits and atom values must be on the same device")
    return manifest


def _edge_weights(
    logits: torch.Tensor, manifest: Sequence[E3Atom], tau: float, mode: str
) -> torch.Tensor:
    if not math.isfinite(float(tau)) or float(tau) <= 0.0:
        raise ValueError("tau must be finite and positive")
    if mode not in {"soft", "hard_st"}:
        raise ValueError("mode must be soft or hard_st")
    probabilities = torch.softmax(logits / float(tau), dim=-1)
    if mode == "soft":
        return probabilities

    atom_count = len(manifest)
    pair_score = logits[:, 0, :, None] + logits[:, 1, None, :]
    diagonal = torch.eye(atom_count, dtype=torch.bool, device=logits.device)
    flattened = pair_score.masked_fill(diagonal[None], -torch.inf).reshape(
        logits.shape[0], -1
    )
    selected = torch.argmax(flattened, dim=1)
    positive = torch.div(selected, atom_count, rounding_mode="floor")
    negative = selected.remainder(atom_count)
    hard = torch.zeros_like(probabilities)
    hard[:, 0].scatter_(1, positive[:, None], 1.0)
    hard[:, 1].scatter_(1, negative[:, None], 1.0)
    return hard - probabilities.detach() + probabilities


def batched_forward(
    logits: torch.Tensor,
    atom_manifest: Sequence[E3Atom],
    atom_values: torch.Tensor,
    *,
    tau: float,
    mode: str,
) -> torch.Tensor:
    """Evaluate M independent categorical miners as a [M,D,I] tensor."""
    manifest = _validate_batch_inputs(logits, atom_manifest, atom_values)
    weights = _edge_weights(logits, manifest, tau, mode)
    positive = torch.einsum("dia,ma->mdi", atom_values, weights[:, 0])
    negative = torch.einsum("dia,ma->mdi", atom_values, weights[:, 1])
    return positive - negative


def batched_training_objective(
    logits: torch.Tensor,
    atom_manifest: Sequence[E3Atom],
    atom_values: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    multiplicity_weights: torch.Tensor,
    *,
    tau: float,
    mode: str,
) -> dict[str, torch.Tensor]:
    """Compute the exact single-miner objective with miner-specific date weights."""
    score = batched_forward(logits, atom_manifest, atom_values, tau=tau, mode=mode)
    if (
        target.shape != tuple(atom_values.shape[:2])
        or target.dtype is not torch.float64
    ):
        raise ValueError("target must be float64 with shape (dates, instruments)")
    if valid_mask.shape != target.shape or valid_mask.dtype is not torch.bool:
        raise ValueError("valid_mask must be bool with the target shape")
    if multiplicity_weights.shape != (logits.shape[0], target.shape[0]):
        raise ValueError("multiplicity weights must have shape (miners, dates)")
    if (
        target.device != logits.device
        or valid_mask.device != logits.device
        or multiplicity_weights.device != logits.device
    ):
        raise ValueError("all objective tensors must use the logits device")
    if not bool(torch.isfinite(target[valid_mask]).all()):
        raise ValueError("target must be finite on the valid mask")
    if not bool(torch.isfinite(multiplicity_weights).all()) or bool(
        (multiplicity_weights < 0).any()
    ):
        raise ValueError("multiplicity weights must be finite and nonnegative")

    counts = valid_mask.sum(dim=1)
    safe_counts = counts.clamp_min(1).to(dtype=torch.float64)
    masked_target = torch.where(valid_mask, target, torch.zeros_like(target))
    target_mean = masked_target.sum(dim=1) / safe_counts
    centered_target = torch.where(
        valid_mask,
        target - target_mean[:, None],
        torch.zeros_like(target),
    )
    expanded_mask = valid_mask[None]
    masked_score = torch.where(expanded_mask, score, torch.zeros_like(score))
    score_mean = masked_score.sum(dim=2) / safe_counts[None]
    centered_score = torch.where(
        expanded_mask,
        score - score_mean[..., None],
        torch.zeros_like(score),
    )
    numerator = torch.sum(centered_score * centered_target[None], dim=2)
    denominator = torch.sqrt(
        torch.sum(centered_score.square(), dim=2)
        * torch.sum(centered_target.square(), dim=1)[None]
    )
    usable = (counts[None] >= 2) & (denominator > 0)
    effective_weights = multiplicity_weights * usable.to(dtype=torch.float64)
    weight_sum = effective_weights.sum(dim=1)
    if bool((weight_sum <= 0).any()):
        raise ValueError("a miner bootstrap has no usable daily cross section")
    correlations = torch.where(usable, numerator / denominator.clamp_min(1e-300), 0.0)
    mean_daily_ic = (correlations * effective_weights).sum(dim=1) / weight_sum

    probabilities = torch.softmax(logits / float(tau), dim=-1)
    entropy = -torch.sum(
        probabilities * torch.log(probabilities.clamp_min(1e-15)), dim=(1, 2)
    )
    edge_overlap = torch.sum(probabilities[:, 0] * probabilities[:, 1], dim=1)
    data_loss = -mean_daily_ic
    total_loss = (
        data_loss
        + ENTROPY_COEFFICIENT * entropy
        + EDGE_OVERLAP_COEFFICIENT * edge_overlap
    )
    return {
        "total_loss": total_loss,
        "data_loss": data_loss,
        "mean_daily_ic": mean_daily_ic,
        "entropy": entropy,
        "edge_overlap": edge_overlap,
        "score": score,
    }


def _initial_logits(profile: str, miner_count: int) -> torch.Tensor:
    profile_index = _profile_index(profile)
    return torch.stack(
        tuple(
            CategoricalE3KAN(
                profile,
                seed=MINER_SEED_BASE + profile_index * 1000 + miner_index,
            ).checkpoint_logits()
            for miner_index in range(miner_count)
        )
    )


def _bootstrap_receipts(
    profile: str, miner_count: int, date_count: int
) -> tuple[BlockBootstrapReceipt, ...]:
    profile_index = _profile_index(profile)
    return tuple(
        draw_training_bootstrap(
            date_count,
            BOOTSTRAP_SEED_BASE + profile_index * MINERS_PER_PROFILE + miner_index,
        )
        for miner_index in range(miner_count)
    )


def _train_batch(
    *,
    profile: str,
    atom_values: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    prediction_mask: torch.Tensor,
    miner_count: int,
    steps: int,
    device: str | torch.device,
    initial_logits: torch.Tensor | None = None,
) -> ProfileRun:
    manifest = build_profile_atom_bank(profile)
    target_device = torch.device(device)
    atoms = atom_values.detach().to(device=target_device, dtype=torch.float64)
    outcomes = target.detach().to(device=target_device, dtype=torch.float64)
    mask = valid_mask.detach().to(device=target_device, dtype=torch.bool)
    fidelity_mask = prediction_mask.detach().to(device=target_device, dtype=torch.bool)
    starting = (
        _initial_logits(profile, miner_count)
        if initial_logits is None
        else initial_logits.detach().to(device="cpu", dtype=torch.float64).clone()
    )
    expected_shape = (miner_count, 2, len(manifest))
    if tuple(starting.shape) != expected_shape:
        raise ValueError(f"initial logits must have shape {expected_shape}")
    if not bool(torch.isfinite(starting).all()):
        raise ValueError("initial logits must be finite")
    logits = torch.nn.Parameter(starting.to(device=target_device).clone())
    bootstraps = _bootstrap_receipts(profile, miner_count, atoms.shape[0])
    weights = torch.tensor(
        [receipt.multiplicities for receipt in bootstraps],
        dtype=torch.float64,
        device=target_device,
    )
    optimizer = torch.optim.Adam((logits,), lr=LEARNING_RATE)
    first_data_gradient: torch.Tensor | None = None
    trajectory: list[_BatchStep] = []
    for update_index in range(steps):
        tau = temperature_at_step(update_index)
        mode = forward_mode_at_step(update_index)
        optimizer.zero_grad(set_to_none=True)
        terms = batched_training_objective(
            logits,
            manifest,
            atoms,
            outcomes,
            mask,
            weights,
            tau=tau,
            mode=mode,
        )
        if update_index == 0:
            first_data_gradient = (
                torch.autograd.grad(
                    terms["data_loss"].sum(), logits, retain_graph=True
                )[0]
                .detach()
                .to(device="cpu")
                .clone()
            )
        terms["total_loss"].sum().backward()
        if logits.grad is None or not bool(torch.isfinite(logits.grad).all()):
            raise FloatingPointError("E3 gate gradient is missing or non-finite")
        optimizer.step()
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("E3 gate update produced non-finite logits")
        trajectory.append(
            _BatchStep(
                update_index=update_index,
                tau=tau,
                mode=mode,
                total_loss=terms["total_loss"].detach().to(device="cpu").clone(),
                mean_daily_ic=terms["mean_daily_ic"].detach().to(device="cpu").clone(),
                entropy=terms["entropy"].detach().to(device="cpu").clone(),
                edge_overlap=terms["edge_overlap"].detach().to(device="cpu").clone(),
                gate_logits=logits.detach().to(device="cpu").clone(),
            )
        )

    final_logits = logits.detach().to(device="cpu").clone()
    if first_data_gradient is None:
        first_data_gradient = torch.zeros_like(final_logits)
    with torch.no_grad():
        soft_predictions = batched_forward(
            logits,
            manifest,
            atoms,
            tau=FINAL_TEMPERATURE,
            mode="soft",
        )
    profile_index = _profile_index(profile)
    miner_receipts: list[MinerReceipt] = []
    for miner_index in range(miner_count):
        hardening = harden_checkpoint(
            final_logits[miner_index], manifest, FINAL_TEMPERATURE
        )
        hard_prediction = evaluate_hard_ast_from_atoms(hardening.ast, manifest, atoms)
        fidelity = soft_hard_fidelity(
            soft_predictions[miner_index][fidelity_mask],
            hard_prediction[fidelity_mask],
        )
        failures: list[str] = []
        if fidelity["pearson"] < FIDELITY_PEARSON_MINIMUM:
            failures.append("low_soft_hard_pearson")
        if fidelity["nrmse"] > FIDELITY_NRMSE_MAXIMUM:
            failures.append("high_soft_hard_nrmse")
        if (
            min(hardening.positive.margin, hardening.negative.margin)
            < GATE_MARGIN_MINIMUM
        ):
            failures.append("low_gate_margin")
        if steps == 0:
            failures.append("test_only_no_updates")
        global_attempt_index = profile_index * MINERS_PER_PROFILE + miner_index
        miner_receipts.append(
            MinerReceipt(
                profile=profile,
                profile_index=profile_index,
                miner_index=miner_index,
                global_attempt_index=global_attempt_index,
                miner_seed=MINER_SEED_BASE + profile_index * 1000 + miner_index,
                bootstrap=bootstraps[miner_index],
                initial_logits=starting[miner_index].clone(),
                final_logits=final_logits[miner_index].clone(),
                first_step_data_gradient=first_data_gradient[miner_index].clone(),
                trajectory=tuple(
                    TrainingStepReceipt(
                        update_index=step.update_index,
                        tau=step.tau,
                        mode=step.mode,
                        total_loss=float(step.total_loss[miner_index]),
                        mean_daily_ic=float(step.mean_daily_ic[miner_index]),
                        entropy=float(step.entropy[miner_index]),
                        edge_overlap=float(step.edge_overlap[miner_index]),
                        gate_logits=step.gate_logits[miner_index].clone(),
                    )
                    for step in trajectory
                ),
                hardening=hardening,
                candidate_ast=hardening.ast,
                fidelity=fidelity,
                admission_failures=tuple(failures),
            )
        )
    return ProfileRun(
        profile=profile,
        device=str(target_device),
        miners=tuple(miner_receipts),
    )


def _training_tensors(
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
    if np.count_nonzero(prediction_mask) < 2:
        raise ValueError("fewer than two jointly supported prediction rows")
    # Unsupported entries are computational sentinels only. Every objective and
    # fidelity consumer receives the exact joint-support mask, so these zeros are
    # never treated as inferred observations or used atom-by-atom.
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


def run_e3_profile(
    panel: PitPanel,
    target: pd.Series,
    profile: str,
    device: str | torch.device = "cpu",
) -> ProfileRun:
    """Run exactly 64 independently seeded, 300-update miners for one profile."""
    maximum_date = pd.to_datetime(panel.raw.index.get_level_values("datetime")).max()
    if maximum_date > pd.Timestamp("2020-12-31"):
        raise ValueError(
            "E3 production training panel crosses the frozen train boundary"
        )
    atom_panel = materialize_atom_panel(panel, profile)
    atoms, outcomes, valid_mask, prediction_mask = _training_tensors(atom_panel, target)
    return _train_batch(
        profile=profile,
        atom_values=atoms,
        target=outcomes,
        valid_mask=valid_mask,
        prediction_mask=prediction_mask,
        miner_count=MINERS_PER_PROFILE,
        steps=TRAINING_STEPS,
        device=device,
    )


def _run_tiny_batch_for_test(
    *,
    profile: str,
    atom_values: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    miner_count: int,
    steps: int,
    device: str | torch.device,
    initial_logits: torch.Tensor | None = None,
) -> ProfileRun:
    """Exercise runner mechanics at reduced scale; never called by production."""
    if not 0 <= steps <= TRAINING_STEPS:
        raise ValueError("test steps must be in [0, 300]")
    if not 1 <= miner_count <= MINERS_PER_PROFILE:
        raise ValueError("test miner_count must be in [1, 64]")
    return _train_batch(
        profile=profile,
        atom_values=atom_values,
        target=target,
        valid_mask=valid_mask,
        prediction_mask=valid_mask,
        miner_count=miner_count,
        steps=steps,
        device=device,
        initial_logits=initial_logits,
    )


__all__ = [
    "AtomPanel",
    "BlockBootstrapReceipt",
    "MinerReceipt",
    "ProfileRun",
    "TrainingStepReceipt",
    "batched_forward",
    "batched_training_objective",
    "draw_training_bootstrap",
    "materialize_atom_panel",
    "run_e3_profile",
]
