"""Single locked data-opening path for the S2a v2 mining pipeline."""

from __future__ import annotations

import json
import hashlib
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import yaml

from mirage_kan.data import PitPanel
from mirage_kan.data.pit import RAW_FIELDS, sha256_file
from mirage_kan.governance.authority import AuthorityGuard, AuthoritySuperseded
from mirage_kan.protocol import BASE_LOCK
from mirage_kan.artifacts.topology import TopologyTransaction

FWD_DEFINITION = "Ref($close, -2) / Ref($close, -1) - 1"


@dataclass(frozen=True)
class LockedMiningInputs:
    """Identity-checked train and validation data after entitlement consumption."""

    panel: PitPanel
    labels: pd.Series
    train: tuple[str, str]
    validation: tuple[str, str]
    cache_sha256: str
    entitlement_sha256: str
    protocol: Mapping[str, object]


class _MiningDataCapability:
    """Non-serializable proof that this process consumed the live opening."""

    __slots__ = ("topology", "authority_guard", "authority_capability", "record")

    def __init__(
        self,
        topology: TopologyTransaction,
        authority_guard: AuthorityGuard,
        authority_capability: str,
        record: Mapping[str, object],
    ) -> None:
        self.topology = topology
        self.authority_guard = authority_guard
        self.authority_capability = authority_capability
        self.record = dict(record)


@dataclass(frozen=True)
class MiningComputation:
    """Complete pre-development output of every frozen mining/control search."""

    kan_runs: tuple[object, ...]
    kan_scoring: object
    kan_selection: object
    gp_generation: object
    gp_scoring: object
    gp_selection: object
    permutation_runs: tuple[object, ...]
    permutation_scoring: object
    permutation_selection: object
    mlp_controls: object
    mechanism_cards: Mapping[str, Mapping[str, object]]
    blind_package: Mapping[str, object]


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _regular_file(root: Path, raw_path: object, label: str) -> Path:
    if not isinstance(raw_path, (str, Path)) or not str(raw_path):
        raise ValueError(f"{label} path is missing")
    path = Path(raw_path)
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"{label} is not a regular file")
    if not path.is_absolute() and root not in resolved.parents:
        raise ValueError(f"{label} escapes the workspace")
    return resolved


def _load_json(path: Path, label: str) -> Mapping[str, object]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), label)


def _load_protocol(
    root: Path,
) -> tuple[Mapping[str, object], Mapping[str, object], Path]:
    lock_path = _regular_file(root, BASE_LOCK, "base lock")
    lock = _load_json(lock_path, "base lock")
    protocol = _mapping(lock.get("protocol"), "base-lock protocol")
    config_path = _regular_file(root, protocol.get("path"), "protocol config")
    if sha256_file(config_path) != protocol.get("sha256"):
        raise ValueError("protocol config hash differs from the base lock")
    config = _mapping(
        yaml.safe_load(config_path.read_text(encoding="utf-8")), "protocol config"
    )
    if config.get("protocol_id") != protocol.get("protocol_id"):
        raise ValueError("protocol ID differs between config and base lock")
    return lock, config, lock_path


def _verify_entitlement(
    root: Path,
    lock: Mapping[str, object],
    config: Mapping[str, object],
    lock_path: Path,
    capability: _MiningDataCapability,
) -> tuple[Mapping[str, object], str]:
    if not isinstance(capability, _MiningDataCapability):
        raise TypeError("mining data require the live in-process opening capability")
    topology = capability.topology
    authority_guard = capability.authority_guard
    if (
        not isinstance(topology, TopologyTransaction)
        or topology.phase != "mining"
        or topology.workspace != root
        or not isinstance(authority_guard, AuthorityGuard)
        or authority_guard.workspace != root
    ):
        raise ValueError("mining data capability belongs to another live execution")
    authority = authority_guard.verify_capability(
        capability.authority_capability, boundary="before_first_label_access"
    )
    if not topology.preclaim_path.is_file() or topology.preclaim_path.is_symlink():
        raise ValueError("mining topology preclaim is missing")
    preclaim_sha256 = sha256_file(topology.preclaim_path)
    for key, target in topology.targets.items():
        marker = target / ".INCOMPLETE"
        if not marker.is_file() or marker.is_symlink():
            raise ValueError(f"mining topology claim is missing: {key}")
        claim = _load_json(marker, f"mining topology claim {key}")
        if (
            claim.get("topology_sha256") != topology.topology_sha256
            or claim.get("topology_key") != key
        ):
            raise ValueError(f"mining topology claim identity mismatch: {key}")
    artifact_paths = _mapping(config.get("artifact_paths"), "artifact paths")
    entitlement_path = _regular_file(
        root, artifact_paths.get("mining_entitlement"), "mining entitlement"
    )
    implementation_path = _regular_file(
        root, artifact_paths.get("implementation_lock"), "implementation lock"
    )
    protocol = _mapping(lock.get("protocol"), "base-lock protocol")
    entitlement = _load_json(entitlement_path, "mining entitlement")
    expected = {
        "schema_version": "mirage_mining_entitlement_v2",
        "protocol_id": protocol.get("protocol_id"),
        "state": "consumed_before_first_label_access",
        "topology_sha256": topology.topology_sha256,
        "topology_preclaim_sha256": preclaim_sha256,
        "authority_receipt_sha256": authority.receipt_sha256,
        "base_lock_sha256": sha256_file(lock_path),
        "config_sha256": protocol.get("sha256"),
        "implementation_lock_sha256": sha256_file(implementation_path),
    }
    for key, value in expected.items():
        if entitlement.get(key) != value:
            raise ValueError(f"mining entitlement identity mismatch: {key}")
    if entitlement != capability.record:
        raise ValueError("mining entitlement differs from the consumed live record")
    return entitlement, sha256_file(entitlement_path)


def _period(config: Mapping[str, object], key: str) -> tuple[str, str]:
    data = _mapping(config.get("data"), "protocol data")
    raw = data.get(key)
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError(f"protocol {key} period is invalid")
    start, end = (str(raw[0]), str(raw[1]))
    if pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError(f"protocol {key} period is reversed")
    return start, end


def _verify_exact_fwd(
    frame: pd.DataFrame,
    periods: tuple[tuple[str, str], tuple[str, str]],
) -> pd.Series:
    indexed = frame.set_index(["datetime", "instrument"]).sort_index()
    observed_raw = indexed["fwd"]
    verified = pd.Series(np.nan, index=indexed.index, dtype=float, name="fwd")
    dates = indexed.index.get_level_values("datetime")
    for start, end in periods:
        in_period = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        period_dates = pd.DatetimeIndex(dates[in_period].unique()).sort_values()
        if len(period_dates) <= 2:
            raise ValueError("mining split has no dates after label-horizon purge")
        allowed_dates = period_dates[:-2]
        period_close = indexed.loc[in_period, "close"]
        by_instrument = period_close.groupby(level="instrument", sort=False)
        expected = (by_instrument.shift(-2) / by_instrument.shift(-1) - 1.0).astype(
            observed_raw.dtype
        )
        expected_finite = (
            np.isfinite(expected.to_numpy(dtype=float))
            & expected.index.get_level_values("datetime").isin(allowed_dates)
            & indexed["in_universe"].reindex(expected.index).to_numpy(dtype=bool)
        )
        observed_period = observed_raw.reindex(expected.index)
        observed_finite = np.isfinite(observed_period.to_numpy(dtype=float))
        if not expected_finite.any() or not np.all(observed_finite[expected_finite]):
            raise ValueError(
                "cache fwd finite support differs from the exact definition"
            )
        if not np.array_equal(
            observed_period.to_numpy(dtype=float)[expected_finite],
            expected.to_numpy(dtype=float)[expected_finite],
            equal_nan=False,
        ):
            raise ValueError("cache fwd values do not match the exact fwd definition")
        verified.loc[expected.index[expected_finite]] = observed_period.loc[
            expected.index[expected_finite]
        ].to_numpy(dtype=float)
    return verified


def _load_locked_mining_inputs(
    workspace: Path | str, capability: _MiningDataCapability
) -> LockedMiningInputs:
    """Load no development rows and only after a live entitlement is present."""
    root = Path(workspace).resolve(strict=True)
    lock, config, lock_path = _load_protocol(root)
    _, entitlement_sha256 = _verify_entitlement(
        root, lock, config, lock_path, capability
    )
    train = _period(config, "train")
    validation = _period(config, "validation")
    if pd.Timestamp(train[1]) >= pd.Timestamp(validation[0]):
        raise ValueError("train and validation periods overlap")
    data = _mapping(config.get("data"), "protocol data")
    if data.get("labels") != ["fwd"] or data.get("raw_miner_inputs") != list(
        RAW_FIELDS
    ):
        raise ValueError("protocol mining inputs differ from raw OHLCV plus fwd")
    purge = _mapping(data.get("label_horizon_purge"), "label-horizon purge")
    if purge != {
        "trading_dates": 2,
        "boundaries": ["train_end", "validation_end"],
        "cross_split_labels_forbidden": True,
    }:
        raise ValueError("protocol label-horizon purge differs from the exact rule")
    warmup = _mapping(data.get("feature_warmup"), "feature warm-up")
    if warmup != {
        "trading_dates": 60,
        "raw_only": True,
        "labels_outside_objective_split_are_null": True,
    }:
        raise ValueError("protocol feature warm-up differs from the exact rule")

    data_lock = _mapping(lock.get("data"), "base-lock data")
    data_config_path = _regular_file(root, data_lock.get("config_path"), "data config")
    if sha256_file(data_config_path) != data_lock.get("config_sha256"):
        raise ValueError("data config hash differs from the base lock")
    data_config = _load_json(data_config_path, "data config")
    if data_config.get("label_1d") != FWD_DEFINITION:
        raise ValueError("data config does not specify the exact fwd definition")
    cache_path = _regular_file(root, data_config.get("cache_path"), "PIT cache")
    cache_sha256 = sha256_file(cache_path)
    if (
        cache_sha256 != data_config.get("cache_sha256")
        or cache_sha256 != data_lock.get("cache_sha256")
        or str(cache_path) != str(Path(str(data_lock.get("cache_path"))).resolve())
    ):
        raise ValueError("PIT cache identity differs from the frozen lock chain")

    raw_columns = ["datetime", "instrument", *RAW_FIELDS, "in_universe"]
    raw_filters = [("datetime", "<=", pd.Timestamp(validation[1]))]
    raw_frame = pd.read_parquet(cache_path, columns=raw_columns, filters=raw_filters)
    if list(raw_frame.columns) != raw_columns or raw_frame.empty:
        raise ValueError("raw PIT cache has an invalid mining schema")
    raw_frame["datetime"] = pd.to_datetime(raw_frame["datetime"])
    available_dates = pd.DatetimeIndex(raw_frame["datetime"].unique()).sort_values()
    preceding_dates = available_dates[available_dates < pd.Timestamp(train[0])][-60:]
    if len(preceding_dates) != 60:
        raise ValueError("PIT cache lacks the exact 60-date feature warm-up")
    mining_start = preceding_dates[0]
    raw_frame = raw_frame.loc[raw_frame["datetime"] >= mining_start].copy()

    label_columns = ["datetime", "instrument", "fwd"]
    allowed_periods: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for start, end in (train, validation):
        period_dates = available_dates[
            (available_dates >= pd.Timestamp(start))
            & (available_dates <= pd.Timestamp(end))
        ]
        if len(period_dates) <= 2:
            raise ValueError("mining split has no dates after label-horizon purge")
        allowed_periods.append((pd.Timestamp(start), period_dates[-3]))
    label_filters = [
        [
            ("datetime", ">=", start),
            ("datetime", "<=", allowed_end),
        ]
        for start, allowed_end in allowed_periods
    ]
    label_frame = pd.read_parquet(
        cache_path, columns=label_columns, filters=label_filters
    )
    if list(label_frame.columns) != label_columns or label_frame.empty:
        raise ValueError("filtered PIT labels have an invalid mining schema")
    frame = raw_frame.loc[
        raw_frame["datetime"].between(
            pd.Timestamp(train[0]), pd.Timestamp(validation[1])
        )
    ].merge(
        label_frame,
        on=["datetime", "instrument"],
        how="left",
        validate="one_to_one",
    )
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    if frame.duplicated(["datetime", "instrument"]).any():
        raise ValueError("filtered PIT cache has duplicate panel rows")
    dates = frame["datetime"]
    if dates.min() < pd.Timestamp(train[0]) or dates.max() > pd.Timestamp(
        validation[1]
    ):
        raise ValueError("filtered PIT cache exposed rows outside train and validation")
    train_rows = dates.between(pd.Timestamp(train[0]), pd.Timestamp(train[1]))
    validation_rows = dates.between(
        pd.Timestamp(validation[0]), pd.Timestamp(validation[1])
    )
    if (
        not train_rows.any()
        or not validation_rows.any()
        or not (train_rows | validation_rows).all()
    ):
        raise ValueError(
            "filtered PIT cache does not exactly partition into train and validation"
        )
    labels = _verify_exact_fwd(frame, (train, validation))
    panel = PitPanel.from_frame(
        raw_frame,
        source_path=cache_path,
        source_sha256=cache_sha256,
    )
    labels = labels.reindex(panel.raw.index)
    return LockedMiningInputs(
        panel=panel,
        labels=labels,
        train=train,
        validation=validation,
        cache_sha256=cache_sha256,
        entitlement_sha256=entitlement_sha256,
        protocol=dict(config),
    )


def _score_candidates(
    candidates, inputs: LockedMiningInputs, labels: pd.Series, *, label_mode: str
):
    from mirage_kan.mining.v2_scoring import score_hard_ast_candidates

    admission = _mapping(inputs.protocol.get("admission"), "admission protocol")
    fidelity = _mapping(
        _mapping(inputs.protocol.get("kan_e3"), "KAN protocol").get("fidelity"),
        "KAN fidelity protocol",
    )
    return score_hard_ast_candidates(
        candidates,
        inputs.panel,
        labels,
        train=inputs.train,
        validation=inputs.validation,
        minimum_coverage=float(admission["minimum_coverage"]),
        minimum_absolute_train_rank_ic=float(
            admission["minimum_absolute_train_rank_ic"]
        ),
        minimum_absolute_validation_rank_ic=float(
            admission["minimum_absolute_validation_rank_ic"]
        ),
        minimum_soft_hard_pearson=float(fidelity["soft_hard_pearson_minimum"]),
        maximum_soft_hard_nrmse=float(fidelity["soft_hard_nrmse_maximum"]),
        minimum_gate_probability_margin=float(
            fidelity["gate_probability_margin_minimum"]
        ),
        label_mode=label_mode,
    )


def _receipt_sha256(receipt: object) -> str:
    def choice(value: object) -> dict[str, object]:
        return dict(vars(value))

    payload = {
        "ast": receipt.ast.to_dict(),
        "positive": choice(receipt.positive),
        "negative": choice(receipt.negative),
        "rejected_alternates": [choice(item) for item in receipt.rejected_alternates],
        "tau": receipt.tau,
        "checkpoint_logits_sha256": receipt.checkpoint_logits_sha256,
        "atom_manifest_sha256": receipt.atom_manifest_sha256,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _selected_programs(selection: object) -> dict[str, object]:
    programs: dict[str, object] = {}
    for candidate in selection.selected:
        if candidate.ast is None or candidate.canonical_hash != candidate.ast.identity:
            raise ValueError("selected factor lacks an exact canonical AST")
        programs[candidate.candidate_id] = candidate.ast
    return programs


def _atom_panel_period(atom_panel: object, bounds: tuple[str, str]):
    """Keep warm-up computation out of period-only mechanism statistics."""
    from mirage_kan.mining.e3_runner import AtomPanel

    if not isinstance(atom_panel, AtomPanel):
        raise TypeError("mechanism slicing requires an AtomPanel")
    start, end = map(pd.Timestamp, bounds)
    keep = (atom_panel.dates >= start) & (atom_panel.dates <= end)
    dates = atom_panel.dates[keep]
    if dates.empty:
        raise ValueError("mechanism period has no atom-panel dates")
    index = pd.MultiIndex.from_product(
        [dates, atom_panel.instruments], names=["datetime", "instrument"]
    )
    return AtomPanel(
        profile=atom_panel.profile,
        atom_manifest=atom_panel.atom_manifest,
        dates=dates,
        instruments=atom_panel.instruments,
        index=index,
        values=atom_panel.values[keep].copy(),
        support=atom_panel.support[keep].copy(),
        joint_support=atom_panel.joint_support[keep].copy(),
        membership=atom_panel.membership[keep].copy(),
    )


def _compute_mining(
    inputs: LockedMiningInputs,
    authority_guard: object,
    *,
    devices: tuple[str, str],
) -> MiningComputation:
    from mirage_kan.artifacts.mechanism import (
        build_blind_review_package,
        build_mechanism_card,
        compute_mechanism_interventions,
    )
    from mirage_kan.governance.authority import AuthorityGuard
    from mirage_kan.mining.e3 import PROFILE_SPECS
    from mirage_kan.mining.e3_runner import materialize_atom_panel, run_e3_profile
    from mirage_kan.mining.gp_control import generate_gp_attempts
    from mirage_kan.mining.mlp_control import (
        MLPControlPairing,
        run_matched_blackbox_controls,
    )
    from mirage_kan.mining.v2_scoring import (
        select_production_candidates,
        select_size_matched_gp_control,
        select_size_matched_null,
    )
    from mirage_kan.mining.v2_workflow import (
        TrainOnlyRankIcScorer,
        gp_attempt_candidates,
        permute_labels_within_membership,
        profile_run_candidates,
        slice_panel_and_labels,
    )

    if not isinstance(authority_guard, AuthorityGuard):
        raise TypeError("mining computation requires the live AuthorityGuard")
    if len(devices) != 2 or len(set(devices)) != 2:
        raise ValueError("production mining requires two distinct compute devices")
    train_panel, train_labels = slice_panel_and_labels(
        inputs.panel, inputs.labels, inputs.train
    )
    controls = _mapping(inputs.protocol.get("controls"), "controls protocol")
    permutation = _mapping(controls.get("label_permutation"), "permutation protocol")
    permuted_labels = permute_labels_within_membership(
        inputs.labels, inputs.panel, seed=int(permutation["permutation_seed"])
    )
    _, permuted_train_labels = slice_panel_and_labels(
        inputs.panel, permuted_labels, inputs.train
    )

    for arm in (
        "kan_e3_selected",
        "typed_gp_sr_control",
        "kan_e3_permutation_control",
    ):
        receipt = authority_guard.revalidate(
            "before_each_scientific_or_control_arm", arm=arm
        )
        authority_guard.verify_capability(
            receipt.capability,
            boundary="before_each_scientific_or_control_arm",
            arm=arm,
        )
    profiles = tuple(PROFILE_SPECS)
    gp_scorer = TrainOnlyRankIcScorer(inputs.panel, inputs.labels, inputs.train)
    with ThreadPoolExecutor(max_workers=9) as executor:
        real_futures = {
            profile: executor.submit(
                run_e3_profile,
                train_panel,
                train_labels,
                profile,
                devices[index % 2],
            )
            for index, profile in enumerate(profiles)
        }
        permutation_futures = {
            profile: executor.submit(
                run_e3_profile,
                train_panel,
                permuted_train_labels,
                profile,
                devices[index % 2],
            )
            for index, profile in enumerate(profiles)
        }
        gp_future = executor.submit(generate_gp_attempts, gp_scorer)
        kan_runs = tuple(real_futures[profile].result() for profile in profiles)
        permutation_runs = tuple(
            permutation_futures[profile].result() for profile in profiles
        )
        gp_generation = gp_future.result()

    with ThreadPoolExecutor(max_workers=4) as executor:
        atom_futures = {
            profile: executor.submit(materialize_atom_panel, inputs.panel, profile)
            for profile in profiles
        }
        atom_panels = {profile: atom_futures[profile].result() for profile in profiles}
    kan_candidates = tuple(
        candidate
        for run in kan_runs
        for candidate in profile_run_candidates(run, atom_panels[run.profile])
    )
    permutation_candidates = tuple(
        candidate
        for run in permutation_runs
        for candidate in profile_run_candidates(run, atom_panels[run.profile])
    )
    kan_scoring = _score_candidates(
        kan_candidates, inputs, inputs.labels, label_mode="real"
    )
    permutation_scoring = _score_candidates(
        permutation_candidates,
        inputs,
        permuted_labels,
        label_mode="within_date_permutation",
    )
    gp_scoring = _score_candidates(
        gp_attempt_candidates(gp_generation), inputs, inputs.labels, label_mode="real"
    )
    admission = _mapping(inputs.protocol.get("admission"), "admission protocol")
    kan_selection = select_production_candidates(
        kan_scoring.candidates,
        library_cap=int(admission["library_cap"]),
        minimum_library_size=int(admission["minimum_library_size"]),
        minimum_profiles=int(admission["minimum_miner_profiles"]),
        maximum_absolute_validation_spearman=float(
            admission["maximum_absolute_validation_spearman"]
        ),
    )
    kan_selection.require_complete()
    selected_count = len(kan_selection.selected)
    gp_selection = select_size_matched_gp_control(
        gp_scoring.candidates,
        target_size=selected_count,
        minimum_library_size=int(admission["minimum_library_size"]),
        library_cap=int(admission["library_cap"]),
        minimum_profiles=int(admission["minimum_miner_profiles"]),
        maximum_absolute_validation_spearman=float(
            admission["maximum_absolute_validation_spearman"]
        ),
    )
    gp_selection.require_complete()
    permutation_selection = select_size_matched_null(
        permutation_scoring.candidates,
        target_size=selected_count,
        minimum_library_size=int(admission["minimum_library_size"]),
        library_cap=int(admission["library_cap"]),
        minimum_profiles=int(admission["minimum_miner_profiles"]),
        maximum_absolute_validation_spearman=float(
            admission["maximum_absolute_validation_spearman"]
        ),
    )
    permutation_selection.require_complete()

    mlp_authority = authority_guard.revalidate(
        "before_each_scientific_or_control_arm", arm="matched_blackbox_control"
    )
    authority_guard.verify_capability(
        mlp_authority.capability,
        boundary="before_each_scientific_or_control_arm",
        arm="matched_blackbox_control",
    )
    miners_by_global = {
        miner.global_attempt_index: miner for run in kan_runs for miner in run.miners
    }
    pairings = tuple(
        MLPControlPairing(
            profile=candidate.profile,
            kan_global_attempt_index=candidate.attempt_index,
            bootstrap=miners_by_global[candidate.attempt_index].bootstrap,
        )
        for candidate in kan_selection.selected
    )
    mlp_controls = run_matched_blackbox_controls(
        train_panel,
        train_labels,
        pairings,
        minimum_library_size=int(admission["minimum_library_size"]),
        library_cap=int(admission["library_cap"]),
        device=devices[0],
    )

    validation_panel, _ = slice_panel_and_labels(
        inputs.panel, inputs.labels, inputs.validation
    )
    validation_atom_panels = {
        profile: _atom_panel_period(
            materialize_atom_panel(validation_panel, profile), inputs.validation
        )
        for profile in {candidate.profile for candidate in kan_selection.selected}
    }
    cards: dict[str, Mapping[str, object]] = {}
    for candidate in kan_selection.selected:
        miner = miners_by_global[candidate.attempt_index]
        atom_panel = validation_atom_panels[candidate.profile]
        variable, lag, local = compute_mechanism_interventions(
            miner.hardening, atom_panel
        )
        cards[candidate.candidate_id] = build_mechanism_card(
            factor_id=candidate.candidate_id,
            profile=candidate.profile,
            receipt=miner.hardening,
            atom_manifest=atom_panel.atom_manifest,
            fidelity={
                "pearson": candidate.fidelity_pearson,
                "nrmse": candidate.fidelity_nrmse,
            },
            variable_interventions=variable,
            lag_interventions=lag,
            local_counterfactual_response=local,
            lineage={
                "miner": "kan_e3",
                "global_seed": miner.miner_seed,
                "checkpoint_sha256": miner.hardening.checkpoint_logits_sha256,
                "hardening_receipt_sha256": _receipt_sha256(miner.hardening),
            },
        )
    review = _mapping(
        _mapping(
            inputs.protocol.get("interpretability"), "interpretability protocol"
        ).get("human_blind_review"),
        "blind-review protocol",
    )
    blind_package = build_blind_review_package(
        cards,
        hides=("method_name", "pnl", "return_metrics"),
        reviewers_minimum=int(review["reviewers_minimum"]),
        response_direction_accuracy_minimum=float(
            review["response_direction_accuracy_minimum"]
        ),
    )
    return MiningComputation(
        kan_runs=kan_runs,
        kan_scoring=kan_scoring,
        kan_selection=kan_selection,
        gp_generation=gp_generation,
        gp_scoring=gp_scoring,
        gp_selection=gp_selection,
        permutation_runs=permutation_runs,
        permutation_scoring=permutation_scoring,
        permutation_selection=permutation_selection,
        mlp_controls=mlp_controls,
        mechanism_cards=cards,
        blind_package=blind_package,
    )


def _staging_path(target: Path) -> Path:
    return target.parent / f".{target.name}.staging"


def _publish_mining(
    topology: object,
    computation: MiningComputation,
    inputs: LockedMiningInputs,
    identities: Mapping[str, str],
    authority_guard: object,
) -> Path:
    from mirage_kan.artifacts.topology import TopologyTransaction
    from mirage_kan.artifacts.v2_bundle import (
        stage_blind_review_package,
        stage_mechanism_cards,
        stage_mining_top_bundle,
        stage_mlp_control_panel,
        stage_v2_factor_library,
    )
    from mirage_kan.governance.authority import AuthorityGuard

    if not isinstance(topology, TopologyTransaction) or not isinstance(
        authority_guard, AuthorityGuard
    ):
        raise TypeError("mining publication requires live topology and authority")
    admission = _mapping(inputs.protocol.get("admission"), "admission protocol")
    minimum_library_size = admission.get("minimum_library_size")
    library_cap = admission.get("library_cap")
    if (
        type(minimum_library_size) is not int
        or type(library_cap) is not int
        or not 1 <= minimum_library_size <= library_cap
    ):
        raise ValueError("admission protocol has invalid library-size bounds")
    selected_ids = [item.candidate_id for item in computation.kan_selection.selected]
    kan_programs = _selected_programs(computation.kan_selection)
    gp_programs = _selected_programs(computation.gp_selection)
    permutation_programs = _selected_programs(computation.permutation_selection)
    lineage = {
        item.candidate_id: {
            "canonical_hash": item.canonical_hash,
            "global_attempt_index": item.attempt_index,
        }
        for item in computation.kan_selection.selected
    }
    builders = {
        "kan_library": lambda path: stage_v2_factor_library(
            path,
            kan_programs,
            inputs.panel,
            topology_key="kan_library",
            identities=identities,
            minimum_library_size=minimum_library_size,
            library_cap=library_cap,
            factor_lineage=lineage,
        ),
        "gp_control_library": lambda path: stage_v2_factor_library(
            path,
            gp_programs,
            inputs.panel,
            topology_key="gp_control_library",
            identities=identities,
            minimum_library_size=minimum_library_size,
            library_cap=library_cap,
        ),
        "permutation_control_library": lambda path: stage_v2_factor_library(
            path,
            permutation_programs,
            inputs.panel,
            topology_key="permutation_control_library",
            identities=identities,
            minimum_library_size=minimum_library_size,
            library_cap=library_cap,
        ),
        "blackbox_control": lambda path: stage_mlp_control_panel(
            path,
            computation.mlp_controls,
            inputs.panel,
            selected_kan_factor_ids=selected_ids,
            identities=identities,
            minimum_library_size=minimum_library_size,
            library_cap=library_cap,
        ),
        "mechanism_cards": lambda path: stage_mechanism_cards(
            path,
            computation.mechanism_cards,
            selected_factor_ids=selected_ids,
            identities=identities,
        ),
        "blind_review_package": lambda path: stage_blind_review_package(
            path,
            computation.blind_package,
            selected_factor_ids=selected_ids,
            hides=("method_name", "pnl", "return_metrics"),
            identities=identities,
        ),
    }
    staging_paths: list[Path] = []
    try:
        for key in topology.child_keys:
            staging = _staging_path(topology.targets[key])
            staging_paths.append(staging)
            bundle = builders[key](staging)
            receipt = authority_guard.revalidate(
                "before_each_artifact_publication", arm=key
            )
            topology.publish_child(
                key,
                bundle.path,
                authority_guard=authority_guard,
                authority_capability=receipt.capability,
            )
        count = len(selected_ids)
        top_staging = _staging_path(topology.targets[topology.top_key])
        staging_paths.append(top_staging)
        top = stage_mining_top_bundle(
            top_staging,
            {key: topology.targets[key] for key in topology.child_keys},
            kan_profile_runs=computation.kan_runs,
            kan_scoring=computation.kan_scoring,
            kan_selection=computation.kan_selection,
            gp_generation=computation.gp_generation,
            gp_scoring=computation.gp_scoring,
            gp_selection=computation.gp_selection,
            permutation_profile_runs=computation.permutation_runs,
            permutation_scoring=computation.permutation_scoring,
            permutation_null_selection=computation.permutation_selection,
            identities=identities,
            budget_counts={
                "kan_attempts": 256,
                "kan_updates": 256 * 300,
                "gp_attempts": 256,
                "permutation_attempts": 256,
                "mlp_controls": count,
                "mlp_updates": count * 300,
            },
            minimum_library_size=minimum_library_size,
            library_cap=library_cap,
        )
        receipt = authority_guard.revalidate(
            "before_each_artifact_publication", arm=topology.top_key
        )
        topology.publish_top_bundle(
            top.path,
            authority_guard=authority_guard,
            authority_capability=receipt.capability,
        )
    finally:
        for staging in staging_paths:
            if staging.exists() and staging.is_dir() and not staging.is_symlink():
                shutil.rmtree(staging)
    return topology.targets[topology.top_key]


def run_s2a_v2_mining(
    workspace: Path | str,
    devices: tuple[str, str] = ("cuda:0", "cuda:1"),
) -> Path:
    """Consume the one mining opening, run every arm, and publish atomically."""
    from mirage_kan.governance.implementation_lock import verify_implementation_lock
    from mirage_kan.governance.openings import consume_mining_entitlement

    root = Path(workspace).resolve(strict=True)
    _, config, _ = _load_protocol(root)
    mining_source = config.get("mining_source")
    if isinstance(mining_source, Mapping) and mining_source.get("mode") == (
        "verified_cross_protocol_rebind"
    ):
        raise RuntimeError(
            "rebind protocol forbids a new mining preclaim or label entitlement"
        )
    implementation = verify_implementation_lock(root)
    topology = TopologyTransaction.from_frozen_config(root, phase="mining")
    topology.preclaim()
    topology.claim_all()
    try:
        guard = AuthorityGuard(root)
        first_label = guard.revalidate("before_first_label_access")
        entitlement = consume_mining_entitlement(
            root, topology, guard, first_label.capability
        )
        capability = _MiningDataCapability(
            topology, guard, first_label.capability, entitlement
        )
        inputs = _load_locked_mining_inputs(root, capability)
        computation = _compute_mining(inputs, guard, devices=devices)
        identities = {
            "protocol_sha256": str(entitlement["config_sha256"]),
            "authority_sha256": first_label.authority_sha256,
            "implementation_sha256": str(entitlement["implementation_lock_sha256"]),
        }
        if implementation.get("protocol_id") != entitlement["protocol_id"]:
            raise ValueError("implementation lock protocol differs from entitlement")
        return _publish_mining(topology, computation, inputs, identities, guard)
    except AuthoritySuperseded as error:
        topology.terminalize(
            {"failure_class": "superseded_authority", "error": str(error)}
        )
        raise
    except BaseException as error:
        topology.terminalize(
            {"failure_class": "s2a_v2_mining_failure", "error": str(error)}
        )
        raise


__all__ = [
    "LockedMiningInputs",
    "MiningComputation",
    "run_s2a_v2_mining",
]
