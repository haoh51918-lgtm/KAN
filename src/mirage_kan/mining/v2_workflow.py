"""Label-isolated GP scoring and exact method-output adapters for S2a v2."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from mirage_kan.data import PitPanel
from mirage_kan.dsl import AstNode, evaluate
from mirage_kan.mining.e3 import atom_manifest_sha256
from mirage_kan.mining.e3_runner import (
    TRAINING_STEPS,
    AtomPanel,
    ProfileRun,
    batched_forward,
)
from mirage_kan.mining.gp_control import GpGenerationResult
from mirage_kan.mining.v2_scoring import HardAstCandidate


def _mean_daily_spearman(left: pd.Series, right: pd.Series) -> float:
    frame = pd.concat({"left": left, "right": right}, axis=1).dropna()
    correlations: list[float] = []
    for _, daily in frame.groupby(level="datetime", sort=False):
        if (
            len(daily) < 2
            or daily["left"].nunique() < 2
            or daily["right"].nunique() < 2
        ):
            continue
        correlation = daily["left"].corr(daily["right"], method="spearman")
        if np.isfinite(correlation):
            correlations.append(float(correlation))
    if not correlations:
        raise ValueError("train-only GP formula has no usable daily RankIC")
    return float(np.mean(correlations))


def slice_panel_and_labels(
    panel: PitPanel,
    labels: pd.Series,
    bounds: tuple[str, str],
) -> tuple[PitPanel, pd.Series]:
    """Keep 60 raw lookback dates while exposing labels only inside the split."""
    start, end = map(pd.Timestamp, bounds)
    if start > end:
        raise ValueError("period boundaries must be ordered")
    if labels.name != "fwd":
        raise ValueError("period slicing requires the exact fwd label")
    dates = pd.to_datetime(panel.raw.index.get_level_values("datetime"))
    if not ((dates >= start) & (dates <= end)).any():
        raise ValueError("period slice has no panel rows")
    available_dates = pd.DatetimeIndex(dates.unique()).sort_values()
    preceding_dates = available_dates[available_dates < start][-60:]
    raw_start = preceding_dates[0] if len(preceding_dates) else start
    index = panel.raw.index[(dates >= raw_start) & (dates <= end)]
    if index.empty:
        raise ValueError("period slice has no panel rows")
    sliced_labels = labels.reindex(panel.raw.index).loc[index].copy()
    sliced_dates = pd.to_datetime(index.get_level_values("datetime"))
    sliced_labels.loc[(sliced_dates < start) | (sliced_dates > end)] = np.nan
    return (
        PitPanel(
            raw=panel.raw.loc[index].copy(),
            membership=panel.membership.loc[index].copy(),
            observed={
                field: support.loc[index].copy()
                for field, support in panel.observed.items()
            },
            tradability=(
                panel.tradability.loc[index].copy()
                if panel.tradability is not None
                else None
            ),
            source_path=panel.source_path,
            source_sha256=panel.source_sha256,
        ),
        sliced_labels,
    )


class TrainOnlyRankIcScorer:
    """A GP callback with raw warm-up but no post-train rows or labels."""

    def __init__(
        self,
        panel: PitPanel,
        labels: pd.Series,
        train: tuple[str, str],
    ) -> None:
        self.panel, self.labels = slice_panel_and_labels(panel, labels, train)

    def __call__(self, ast: AstNode, context: object) -> float:
        """Score one formula without exposing any validation object or value."""
        del context
        result = evaluate(ast, self.panel)
        values = result.values.where(result.support & self.panel.membership)
        return _mean_daily_spearman(values, self.labels)


def _lineage_complete(run: ProfileRun, miner_index: int) -> bool:
    miner = run.miners[miner_index]
    return bool(
        len(miner.trajectory) == TRAINING_STEPS
        and [step.update_index for step in miner.trajectory]
        == list(range(TRAINING_STEPS))
        and miner.candidate_ast.identity == miner.hardening.ast.identity
        and miner.final_logits.shape == miner.initial_logits.shape
        and miner.first_step_data_gradient.shape == miner.initial_logits.shape
    )


def profile_run_candidates(
    run: ProfileRun, atom_panel: AtomPanel
) -> tuple[HardAstCandidate, ...]:
    """Recompute final soft outputs and adapt every KAN miner to common scoring."""
    if run.profile != atom_panel.profile:
        raise ValueError("profile run and atom panel profile differ")
    if not run.miners:
        raise ValueError("profile run has no miners")
    if [miner.miner_index for miner in run.miners] != list(range(len(run.miners))):
        raise ValueError("profile run miner indices are not contiguous")
    run_manifest = run.miners[0].hardening.atom_manifest_sha256
    if any(
        miner.profile != run.profile
        or miner.hardening.atom_manifest_sha256 != run_manifest
        for miner in run.miners
    ):
        raise ValueError("profile run contains mixed miner or atom-manifest lineage")
    if atom_manifest_sha256(atom_panel.atom_manifest) != run_manifest:
        raise ValueError("profile run atom manifest does not match the atom panel")

    values = np.where(atom_panel.support, atom_panel.values, 0.0)
    logits = torch.stack(tuple(miner.final_logits for miner in run.miners))
    with torch.no_grad():
        soft = batched_forward(
            logits,
            atom_panel.atom_manifest,
            torch.from_numpy(np.array(values, dtype=np.float64, copy=True)),
            tau=0.10,
            mode="soft",
        ).to(device="cpu")
    supported = atom_panel.joint_support & atom_panel.membership
    candidates: list[HardAstCandidate] = []
    for miner_index, miner in enumerate(run.miners):
        flattened = soft[miner_index].numpy().reshape(-1)
        soft_values = pd.Series(flattened, index=atom_panel.index, dtype=float).where(
            supported.reshape(-1)
        )
        candidates.append(
            HardAstCandidate(
                candidate_id=f"kan_{run.profile}_{miner_index:03d}",
                profile=run.profile,
                attempt_index=miner.global_attempt_index,
                ast=miner.candidate_ast,
                method="kan_e3",
                lineage_complete=_lineage_complete(run, miner_index),
                soft_values=soft_values,
                gate_probability_margin=min(
                    miner.hardening.positive.margin,
                    miner.hardening.negative.margin,
                ),
            )
        )
    return tuple(candidates)


def gp_attempt_candidates(
    result: GpGenerationResult,
) -> tuple[HardAstCandidate, ...]:
    """Adapt all GP budget rows, including invalid and duplicate attempts."""
    if [attempt.global_attempt_index for attempt in result.attempts] != list(
        range(len(result.attempts))
    ):
        raise ValueError("GP global attempt indices are not contiguous")
    return tuple(
        HardAstCandidate(
            candidate_id=attempt.candidate_id,
            profile=attempt.profile,
            attempt_index=attempt.global_attempt_index,
            ast=attempt.ast,
            method="typed_gp_sr",
            lineage_complete=True,
        )
        for attempt in result.attempts
    )


def permute_labels_within_membership(
    labels: pd.Series,
    panel: PitPanel,
    *,
    seed: int,
) -> pd.Series:
    """Permute finite labels within each date and frozen universe membership."""
    if labels.name != "fwd" or not isinstance(labels.index, pd.MultiIndex):
        raise ValueError("label permutation requires exact MultiIndex fwd labels")
    aligned = labels.reindex(panel.raw.index)
    random = np.random.Generator(np.random.PCG64(int(seed)))
    permuted = aligned.copy()
    dates = panel.raw.index.get_level_values("datetime")
    membership = panel.membership.to_numpy(dtype=bool)
    finite = np.isfinite(aligned.to_numpy(dtype=float))
    for date in pd.Index(dates).unique():
        positions = np.flatnonzero((dates == date) & membership & finite)
        if len(positions) > 1:
            values = aligned.iloc[positions].to_numpy(copy=True)
            permuted.iloc[positions] = values[random.permutation(len(values))]
    return permuted.rename("fwd")


__all__ = [
    "TrainOnlyRankIcScorer",
    "gp_attempt_candidates",
    "permute_labels_within_membership",
    "profile_run_candidates",
    "slice_panel_and_labels",
]
