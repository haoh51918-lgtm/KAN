"""Assemble the S2a v2 decision only from immutable published evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from mirage_kan.artifacts.topology import TopologyTransaction
from mirage_kan.data.pit import sha256_file
from mirage_kan.dsl import AstNode, DslType, ProgramError
from mirage_kan.evaluation.s2a_v2_decision import ARMS, decide_s2a_v2
from mirage_kan.governance.implementation_lock import verify_implementation_lock
from mirage_kan.governance.mining_rebind import verify_mining_rebind_receipt
from mirage_kan.mining.e3 import build_profile_atom_bank
from mirage_kan.protocol import BASE_LOCK

_MINING_CHILDREN = {
    "kan_library",
    "gp_control_library",
    "permutation_control_library",
    "blackbox_control",
    "mechanism_cards",
    "blind_review_package",
}


@dataclass(frozen=True)
class StagedDecisionArtifact:
    """An authority-neutral decision bundle awaiting topology publication."""

    path: Path
    manifest: dict[str, object]
    decision: dict[str, object]
    evidence: dict[str, object]
    manifest_sha256: str


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _read_json(path: Path, *, label: str) -> Mapping[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    return _mapping(value, label=label)


def _contained(root: Path, raw: object, *, label: str, directory: bool) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} path must be a nonempty string")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} path escapes the workspace")
    candidate = root / relative
    if candidate.is_symlink():
        raise ValueError(f"{label} path is a symlink")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError(f"{label} path escapes the workspace")
    if directory != resolved.is_dir():
        expected = "directory" if directory else "file"
        raise ValueError(f"{label} path is not a {expected}")
    current = root
    for part in resolved.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} path contains a symlink")
    return resolved


def _sha(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a SHA-256 digest")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{label} must be a SHA-256 digest") from error
    return value


def _file_hash(entry: object, *, label: str) -> str:
    if isinstance(entry, Mapping):
        return _sha(entry.get("sha256"), label=label)
    return _sha(entry, label=label)


def _library_size_bounds(config: Mapping[str, object]) -> tuple[int, int]:
    admission = _mapping(config.get("admission"), label="frozen admission")
    minimum = admission.get("minimum_library_size")
    maximum = admission.get("library_cap")
    if (
        type(minimum) is not int
        or type(maximum) is not int
        or not 1 <= minimum <= maximum
    ):
        raise ValueError("frozen admission has invalid library-size bounds")
    decision = _mapping(config.get("s2a_decision"), label="frozen S2a decision")
    integrity = _mapping(decision.get("integrity"), label="frozen decision integrity")
    integrity_minimum = integrity.get("production_library_size_minimum")
    if type(integrity_minimum) is not int or integrity_minimum != minimum:
        raise ValueError(
            "decision integrity library-size minimum differs from admission minimum"
        )
    return minimum, maximum


def _verify_flat_bundle(
    path: Path, manifest: Mapping[str, object], *, label: str
) -> None:
    files = _mapping(manifest.get("files"), label=f"{label} files")
    entries = tuple(path.iterdir())
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ValueError(f"{label} must contain only flat regular files")
    if {entry.name for entry in entries} != set(files) | {"manifest.json"}:
        raise ValueError(f"{label} file inventory differs from its manifest")
    for filename, record in files.items():
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"{label} contains an unsafe filename")
        expected = _file_hash(record, label=f"{label} {filename}")
        file_path = path / filename
        if sha256_file(file_path) != expected:
            raise ValueError(f"{label} file hash mismatch: {filename}")
        if isinstance(record, Mapping):
            byte_count = record.get("bytes")
            if type(byte_count) is not int or byte_count != file_path.stat().st_size:
                raise ValueError(f"{label} byte count mismatch: {filename}")


def _json_lines(
    path: Path, *, label: str, expected_count: int | None = None
) -> list[Mapping[str, object]]:
    try:
        rows = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON Lines") from error
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"{label} must contain exactly {expected_count} rows")
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError(f"{label} rows must be mappings")
    return rows


def _calendar_sha256(calendar: pd.DatetimeIndex) -> str:
    payload = [value.isoformat() for value in calendar]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _information_ratio(values: np.ndarray) -> float:
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("daily excess returns are not a finite vector")
    deviation = float(np.std(values, ddof=1))
    if not math.isfinite(deviation) or deviation <= 0.0:
        raise ValueError("daily excess returns have zero or non-finite variance")
    return math.sqrt(252.0) * float(np.mean(values)) / deviation


def _load_context(workspace: Path | str) -> dict[str, object]:
    root = Path(workspace).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("decision workspace must be a directory")
    lock_path = _contained(root, str(BASE_LOCK), label="base lock", directory=False)
    lock = _read_json(lock_path, label="base lock")
    protocol = _mapping(lock.get("protocol"), label="base lock protocol")
    config_path = _contained(
        root, protocol.get("path"), label="frozen config", directory=False
    )
    config_sha256 = sha256_file(config_path)
    if config_sha256 != protocol.get("sha256"):
        raise ValueError("frozen config hash differs from the base lock")
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError("frozen config is not valid YAML") from error
    config = _mapping(config, label="frozen config")
    if config.get("protocol_id") != protocol.get("protocol_id"):
        raise ValueError("frozen config protocol differs from the base lock")
    artifact_paths = _mapping(config.get("artifact_paths"), label="artifact paths")
    implementation_path = _contained(
        root,
        artifact_paths.get("implementation_lock"),
        label="implementation lock",
        directory=False,
    )
    topology = TopologyTransaction.from_frozen_config(root, phase="development")
    mining_source = config.get("mining_source")
    if isinstance(mining_source, Mapping) and mining_source.get("mode") == (
        "verified_cross_protocol_rebind"
    ):
        mining_binding = verify_mining_rebind_receipt(
            root,
            target_base_lock_path=lock_path,
            target_implementation_lock_path=implementation_path,
        )
    else:
        mining_binding = None
    return {
        "root": root,
        "lock": lock,
        "lock_path": lock_path,
        "config": config,
        "config_path": config_path,
        "config_sha256": config_sha256,
        "implementation_sha256": sha256_file(implementation_path),
        "artifact_paths": artifact_paths,
        "topology": topology,
        "mining_binding": mining_binding,
    }


def _verify_opening(context: Mapping[str, object]) -> tuple[Mapping[str, object], Path]:
    root = context["root"]
    config = context["config"]
    artifact_paths = context["artifact_paths"]
    topology = context["topology"]
    opening_path = _contained(
        root,
        artifact_paths.get("development_opening"),
        label="development opening",
        directory=False,
    )
    opening = _read_json(opening_path, label="development opening")
    mining_binding = context.get("mining_binding")
    expected_schema = (
        "mirage_s2a_development_opening_v3"
        if mining_binding is not None
        else "mirage_s2a_development_opening_v2"
    )
    if (
        opening.get("schema_version") != expected_schema
        or opening.get("protocol_id") != config.get("protocol_id")
        or opening.get("state") != "consumed_before_first_development_access"
        or opening.get("topology_sha256") != topology.topology_sha256
    ):
        raise ValueError("development opening has an invalid identity or state")
    evaluations = _mapping(artifact_paths.get("evaluations"), label="evaluation paths")
    if set(evaluations) != set(ARMS) or opening.get("evaluation_paths") != dict(
        evaluations
    ):
        raise ValueError("development opening does not bind the exact five arms")
    pins = _mapping(opening.get("identity_pins"), label="development opening pins")
    expected_pins = {
        "base_lock_sha256": sha256_file(context["lock_path"]),
        "implementation_lock_sha256": context["implementation_sha256"],
        "mining_manifest_sha256": _sha(
            pins.get("mining_manifest_sha256"), label="opening mining manifest"
        ),
        "provider_identity": pins.get("provider_identity"),
    }
    if mining_binding is not None:
        mining_source = _mapping(
            config.get("mining_source"), label="rebound mining source"
        )
        receipt_path = _contained(
            root,
            mining_source.get("rebind_receipt"),
            label="mining rebind receipt",
            directory=False,
        )
        expected_pins["mining_rebind_receipt_sha256"] = sha256_file(receipt_path)
        if opening.get("mining_authorization_kind") != (
            "verified_cross_protocol_rebind"
        ):
            raise ValueError("development opening lacks rebind authorization")
    if dict(pins) != expected_pins:
        raise ValueError("development opening identity pins are stale or incomplete")
    for key in (
        "development_calendar_sha256",
        "development_calendar_count",
        "development_calendar_start",
        "development_calendar_end",
    ):
        if key not in opening:
            raise ValueError(f"development opening lacks exact calendar field: {key}")
    _sha(opening["development_calendar_sha256"], label="opening calendar")
    return opening, opening_path


def _verify_development_claim(context: Mapping[str, object]) -> None:
    topology = context["topology"]
    preclaim = _read_json(topology.preclaim_path, label="development preclaim")
    if preclaim.get("topology_sha256") != topology.topology_sha256:
        raise ValueError("development topology was not prospectively preclaimed")
    top = topology.targets[topology.top_key]
    marker = _read_json(top / ".INCOMPLETE", label="decision topology claim")
    if (
        marker.get("topology_sha256") != topology.topology_sha256
        or marker.get("topology_key") != topology.top_key
    ):
        raise ValueError("decision topology target is not owned by this transaction")


def _load_evaluations(
    context: Mapping[str, object], opening: Mapping[str, object], opening_path: Path
) -> tuple[
    dict[str, Mapping[str, object]],
    dict[str, pd.DataFrame],
    pd.DatetimeIndex,
    dict[str, str],
]:
    topology = context["topology"]
    opening_sha256 = sha256_file(opening_path)
    opening_pins = _mapping(opening.get("identity_pins"), label="opening pins")
    expected_pins = {
        **dict(opening_pins),
        "development_opening_sha256": opening_sha256,
        "development_topology_sha256": topology.topology_sha256,
    }
    evaluations: dict[str, Mapping[str, object]] = {}
    daily: dict[str, pd.DataFrame] = {}
    hashes: dict[str, str] = {}
    calendar: pd.DatetimeIndex | None = None
    for arm in ARMS:
        path = topology.targets[f"evaluation:{arm}"]
        if (path / ".INCOMPLETE").exists() or (path / "terminal_failure.json").exists():
            raise ValueError(f"development arm is not immutably published: {arm}")
        manifest_path = path / "manifest.json"
        manifest = _read_json(manifest_path, label=f"{arm} evaluation manifest")
        if (
            manifest.get("schema_version") != "mirage_s2a_quanta_evaluation_v2"
            or manifest.get("protocol_id") != context["config"].get("protocol_id")
            or manifest.get("arm") != arm
            or manifest.get("topology_key") != f"evaluation:{arm}"
            or manifest.get("topology_sha256") != topology.topology_sha256
        ):
            raise ValueError(f"{arm} evaluation manifest has invalid ownership")
        if manifest.get("identity_pins") != expected_pins:
            raise ValueError(f"{arm} evaluation identity pins differ from the opening")
        _verify_flat_bundle(path, manifest, label=f"{arm} evaluation")
        diagnostics = _mapping(
            manifest.get("diagnostic_files"), label=f"{arm} diagnostic files"
        )
        portfolio_hash = _sha(
            diagnostics.get("portfolio_daily.parquet"),
            label=f"{arm} portfolio diagnostic",
        )
        portfolio_path = path / "portfolio_daily.parquet"
        if sha256_file(portfolio_path) != portfolio_hash:
            raise ValueError(f"{arm} portfolio diagnostic hash mismatch")
        frame = pd.read_parquet(portfolio_path)
        required = ("daily_excess_return", "turnover", "realized_cost")
        if not set(required).issubset(frame.columns):
            raise ValueError(f"{arm} daily diagnostics lack required columns")
        observed = pd.DatetimeIndex(pd.to_datetime(frame.index))
        if observed.has_duplicates or not observed.is_monotonic_increasing:
            raise ValueError(f"{arm} daily diagnostics have an invalid calendar")
        selected = frame.loc[:, required]
        values = selected.to_numpy(dtype=float)
        if (
            not np.isfinite(values).all()
            or (selected[["turnover", "realized_cost"]] < 0).any().any()
        ):
            raise ValueError(f"{arm} daily diagnostics contain invalid values")
        if calendar is None:
            calendar = observed
        elif not observed.equals(calendar):
            raise ValueError(
                "five-arm daily diagnostics do not share the same calendar"
            )
        metrics = _mapping(manifest.get("metrics"), label=f"{arm} metrics")
        reported_ir = float(metrics.get("information_ratio"))
        recomputed_ir = _information_ratio(
            selected["daily_excess_return"].to_numpy(dtype=float)
        )
        if not math.isclose(reported_ir, recomputed_ir, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                f"{arm} headline information ratio differs from daily recomputation"
            )
        evaluations[arm] = manifest
        daily[arm] = selected
        hashes[arm] = sha256_file(manifest_path)
    assert calendar is not None
    if (
        opening.get("development_calendar_sha256") != _calendar_sha256(calendar)
        or opening.get("development_calendar_count") != len(calendar)
        or opening.get("development_calendar_start") != calendar[0].isoformat()
        or opening.get("development_calendar_end") != calendar[-1].isoformat()
    ):
        raise ValueError(
            "published daily calendar differs from the development opening"
        )
    return evaluations, daily, calendar, hashes


def _selected_scoring_rows(
    rows: Sequence[Mapping[str, object]],
    selection: Mapping[str, object],
    *,
    label: str,
    expected_target_size: int | None,
    minimum_library_size: int,
    library_cap: int,
) -> tuple[list[Mapping[str, object]], int]:
    if len(rows) != 256 or [row.get("attempt_index") for row in rows] != list(
        range(256)
    ):
        raise ValueError(f"{label} does not bind the exact 256-attempt ledger")
    ids = [row.get("candidate_id") for row in rows]
    if (
        any(not isinstance(value, str) or not value for value in ids)
        or len(set(ids)) != 256
    ):
        raise ValueError(f"{label} candidate identities are invalid")
    by_id = {str(row["candidate_id"]): row for row in rows}
    selected_ids = selection.get("selected_candidate_ids")
    dispositions = selection.get("dispositions")
    if not isinstance(selected_ids, list):
        raise ValueError(f"{label} selection record is incomplete or invalid")
    if not minimum_library_size <= len(selected_ids) <= library_cap:
        raise ValueError(f"{label} selection violates frozen admission bounds")
    if (
        any(not isinstance(value, str) or value not in by_id for value in selected_ids)
        or len(set(selected_ids)) != len(selected_ids)
        or not isinstance(dispositions, Mapping)
        or set(dispositions) != set(by_id)
        or selection.get("minimum_size_met") is not True
        or selection.get("exact_size_met") is not True
        or selection.get("profile_quota_met") is not True
        or selection.get("target_size") != expected_target_size
    ):
        raise ValueError(f"{label} selection record is incomplete or invalid")
    selected_set = set(selected_ids)
    for candidate_id, row in by_id.items():
        disposition = dispositions[candidate_id]
        if not isinstance(disposition, str) or not disposition:
            raise ValueError(f"{label} selection disposition is invalid")
        if candidate_id in selected_set:
            if disposition != "selected" or row.get("production_eligible") is not True:
                raise ValueError(f"{label} selected a production-ineligible candidate")
        elif disposition == "selected":
            raise ValueError(f"{label} has an unregistered selected candidate")
        elif row.get("production_eligible") is not True and disposition != row.get(
            "production_disposition"
        ):
            raise ValueError(f"{label} rewrote an ineligible candidate disposition")
    admitted_count = sum(row.get("production_eligible") is True for row in rows)
    return [by_id[candidate_id] for candidate_id in selected_ids], admitted_count


def _effective_rank(
    panel_path: Path,
    selected_ids: Sequence[str],
    validation: Sequence[object],
    *,
    minimum_joint_rows: int,
) -> float:
    panel = pd.read_parquet(panel_path)
    if (
        not isinstance(panel.index, pd.MultiIndex)
        or "datetime" not in panel.index.names
    ):
        raise ValueError("factor panel lacks the frozen datetime multi-index")
    if list(panel.columns) != list(selected_ids):
        raise ValueError("factor panel columns differ from the selected library order")
    dates = pd.DatetimeIndex(pd.to_datetime(panel.index.get_level_values("datetime")))
    start, end = pd.Timestamp(validation[0]), pd.Timestamp(validation[1])
    frame = panel.loc[(dates >= start) & (dates <= end), list(selected_ids)].dropna()
    standardized: list[np.ndarray] = []
    for _, group in frame.groupby(level="datetime", sort=False):
        values = group.to_numpy(dtype=float)
        deviations = values.std(axis=0, ddof=0)
        if np.isfinite(deviations).all() and (deviations > 0.0).all():
            standardized.append((values - values.mean(axis=0)) / deviations)
    if not standardized:
        raise ValueError(
            "effective rank has no jointly finite nonconstant validation rows"
        )
    pooled = np.concatenate(standardized, axis=0)
    if len(pooled) < minimum_joint_rows:
        raise ValueError("effective rank has insufficient joint validation rows")
    if pooled.shape[1] == 1:
        return 1.0
    correlation = np.corrcoef(pooled, rowvar=False)
    if (
        correlation.shape != (len(selected_ids), len(selected_ids))
        or not np.isfinite(correlation).all()
    ):
        raise ValueError("effective-rank correlation matrix is invalid")
    eigenvalues = np.maximum(np.linalg.eigvalsh(correlation), 0.0)
    probabilities = eigenvalues / eigenvalues.sum()
    entropy = -float(
        np.sum(
            probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0])
        )
    )
    result = math.exp(entropy)
    if not math.isfinite(result):
        raise ValueError("effective rank is non-finite")
    return result


def _strict_kan_ast(row: Mapping[str, object], ast: AstNode) -> bool:
    try:
        contract = ast.validate()
    except (ProgramError, ValueError):
        return False
    profile = row.get("profile")
    try:
        bank = {atom.canonical_hash for atom in build_profile_atom_bank(str(profile))}
    except ValueError:
        return False
    return (
        ast.op == "Sub"
        and len(ast.children) == 2
        and ast.children[0].identity in bank
        and ast.children[1].identity in bank
        and ast.children[0].identity != ast.children[1].identity
        and contract.output_type is DslType.DIMENSIONLESS_TS
        and contract.causal
        and row.get("causal") is True
        and row.get("unique") is True
        and row.get("lineage_gate_met") is True
        and row.get("fidelity_gate_met") is True
        and row.get("production_eligible") is True
    )


def _load_mining_evidence(
    context: Mapping[str, object], opening: Mapping[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    root = context["root"]
    config = context["config"]
    minimum_library_size, library_cap = _library_size_bounds(config)
    mining_binding = context.get("mining_binding")
    if mining_binding is None:
        paths = context["artifact_paths"]
        expected_protocol_sha256 = context["config_sha256"]
        expected_implementation_sha256 = context["implementation_sha256"]
    else:
        source = _mapping(
            _mapping(mining_binding, label="mining binding").get("source"),
            label="rebound mining source",
        )
        artifacts = _mapping(
            source.get("artifacts"), label="rebound mining artifacts"
        )
        paths = {
            key: _mapping(value, label=f"rebound artifact {key}").get("path")
            for key, value in artifacts.items()
        }
        expected_protocol_sha256 = _mapping(
            source.get("config"), label="rebound source config"
        ).get("sha256")
        expected_implementation_sha256 = _mapping(
            source.get("implementation_lock"),
            label="rebound source implementation",
        ).get("sha256")
    mining_path = _contained(
        root, paths.get("mining_run"), label="mining top", directory=True
    )
    top_path = mining_path / "manifest.json"
    top = _read_json(top_path, label="mining top manifest")
    opening_pins = _mapping(opening.get("identity_pins"), label="opening pins")
    if sha256_file(top_path) != opening_pins.get("mining_manifest_sha256"):
        raise ValueError("mining top manifest differs from the development opening")
    if (
        top.get("schema_version") != "mirage_s2a_v2_staging_bundle_v1"
        or top.get("role") != "mining_top_bundle"
        or top.get("topology_key") != "mining_run"
        or top.get("published_child_topology_sha256") != top.get("topology_sha256")
    ):
        raise ValueError("mining top manifest has invalid schema or ownership")
    _verify_flat_bundle(mining_path, top, label="mining top")
    identities = _mapping(top.get("identities"), label="mining identities")
    if (
        identities.get("protocol_sha256") != expected_protocol_sha256
        or identities.get("implementation_sha256")
        != expected_implementation_sha256
    ):
        raise ValueError("mining top is stale relative to the frozen implementation")
    _sha(identities.get("authority_sha256"), label="mining authority")
    child_hashes = _mapping(
        top.get("child_manifest_sha256"), label="mining child hashes"
    )
    child_paths = _mapping(top.get("published_child_paths"), label="mining child paths")
    if set(child_hashes) != _MINING_CHILDREN or set(child_paths) != _MINING_CHILDREN:
        raise ValueError("mining top does not bind the exact child topology")
    children: dict[str, tuple[Path, Mapping[str, object]]] = {}
    topology_sha = _sha(top.get("topology_sha256"), label="mining topology")
    for key in sorted(_MINING_CHILDREN):
        path = _contained(root, paths.get(key), label=key, directory=True)
        if Path(str(child_paths[key])).resolve(strict=True) != path:
            raise ValueError(f"mining top publishes the wrong path for {key}")
        manifest_path = path / "manifest.json"
        if sha256_file(manifest_path) != child_hashes[key]:
            raise ValueError(f"mining top child hash mismatch: {key}")
        manifest = _read_json(manifest_path, label=f"{key} manifest")
        if (
            manifest.get("topology_sha256") != topology_sha
            or manifest.get("topology_key") != key
            or manifest.get("identities") != identities
        ):
            raise ValueError(f"mining child ownership differs: {key}")
        _verify_flat_bundle(path, manifest, label=key)
        children[key] = path, manifest

    expected_libraries = {
        "kan_library": ("kan_e3_selected", True),
        "gp_control_library": ("typed_gp_sr_control", False),
        "permutation_control_library": ("kan_e3_permutation_control", False),
    }
    library_ids: dict[str, list[str]] = {}
    for key, (role, kan_mined) in expected_libraries.items():
        _, manifest = children[key]
        factors = _mapping(manifest.get("factors"), label=f"{key} factors")
        selected = manifest.get("selected_candidate_ids")
        factor_ids = list(factors)
        if selected is None:
            selected = factor_ids
        if not isinstance(selected, list):
            raise ValueError(f"{key} factor inventory is invalid")
        if not minimum_library_size <= len(selected) <= library_cap:
            raise ValueError(f"{key} factor inventory violates frozen admission bounds")
        if (
            manifest.get("schema_version") != "mirage_factor_library_v1"
            or manifest.get("library_role") != role
            or manifest.get("kan_mined") is not kan_mined
            or type(manifest.get("factor_count")) is not int
            or manifest.get("factor_count") != len(factors)
            or selected != factor_ids
        ):
            raise ValueError(f"{key} factor inventory is invalid")
        library_ids[key] = selected

    kan_ids = library_ids["kan_library"]
    library_size = len(kan_ids)
    if any(len(library_ids[key]) != library_size for key in library_ids):
        raise ValueError("published factor libraries are not exactly size matched")
    blackbox = children["blackbox_control"][1]
    lineage = _mapping(top.get("kan_selected_lineage"), label="KAN selected lineage")
    blackbox_ids = blackbox.get("selected_kan_factor_ids")
    blackbox_indices = blackbox.get("paired_kan_global_attempt_indices")
    if (
        blackbox.get("schema_version") != "mirage_matched_blackbox_control_v2"
        or blackbox.get("control_count") != library_size
        or not isinstance(blackbox_ids, list)
        or not isinstance(blackbox_indices, list)
        or len(blackbox_ids) != library_size
        or len(blackbox_indices) != library_size
        or set(blackbox_ids) != set(kan_ids)
        or any(
            _mapping(lineage[factor_id], label=f"KAN lineage {factor_id}").get(
                "global_attempt_index"
            )
            != global_index
            for factor_id, global_index in zip(
                blackbox_ids, blackbox_indices, strict=True
            )
        )
    ):
        raise ValueError(
            "blackbox controls do not pair one-to-one with selected KAN factors"
        )

    required_top_files = {
        "kan_real_scoring.jsonl",
        "kan_real_selection.json",
        "gp_scoring.jsonl",
        "gp_selection.json",
        "kan_profile_runs.jsonl",
        "kan_permutation_false_positive_ledger.jsonl",
    }
    if not required_top_files.issubset(
        _mapping(top.get("files"), label="mining files")
    ):
        raise ValueError("mining top lacks decision-source ledgers")
    kan_rows = _json_lines(
        mining_path / "kan_real_scoring.jsonl", label="KAN scoring", expected_count=256
    )
    gp_rows = _json_lines(
        mining_path / "gp_scoring.jsonl", label="GP scoring", expected_count=256
    )
    kan_selection = _read_json(
        mining_path / "kan_real_selection.json", label="KAN selection"
    )
    gp_selection = _read_json(mining_path / "gp_selection.json", label="GP selection")
    profile_rows = _json_lines(
        mining_path / "kan_profile_runs.jsonl",
        label="KAN profile evidence",
        expected_count=256,
    )
    if [row.get("global_attempt_index") for row in profile_rows] != list(range(256)):
        raise ValueError("KAN profile evidence has invalid global-attempt order")
    false_rows = _json_lines(
        mining_path / "kan_permutation_false_positive_ledger.jsonl",
        label="permutation false-positive ledger",
        expected_count=256,
    )
    kan_selected, kan_admitted = _selected_scoring_rows(
        kan_rows,
        kan_selection,
        label="KAN scoring",
        expected_target_size=None,
        minimum_library_size=minimum_library_size,
        library_cap=library_cap,
    )
    gp_selected, gp_admitted = _selected_scoring_rows(
        gp_rows,
        gp_selection,
        label="GP scoring",
        expected_target_size=library_size,
        minimum_library_size=minimum_library_size,
        library_cap=library_cap,
    )
    if {row["candidate_id"] for row in kan_selected} != set(kan_ids):
        raise ValueError("KAN scoring selection differs from the published library")
    if {row["candidate_id"] for row in gp_selected} != set(
        library_ids["gp_control_library"]
    ):
        raise ValueError("GP scoring selection differs from the published library")
    if [row.get("global_attempt_index") for row in false_rows] != list(range(256)):
        raise ValueError("permutation false-positive ledger has invalid attempt order")
    false_positive_count = sum(
        row.get("production_eligible") is True for row in false_rows
    )
    permutation_ledger = _mapping(
        top.get("permutation_ledger"), label="permutation ledger"
    )
    if (
        permutation_ledger.get("real_threshold_false_positive_count")
        != false_positive_count
        or permutation_ledger.get("real_threshold_false_positive_rows") != 256
    ):
        raise ValueError("permutation false-positive summary differs from its ledger")

    factors = _mapping(children["kan_library"][1].get("factors"), label="KAN factors")
    profile_by_index = {row.get("global_attempt_index"): row for row in profile_rows}
    score_by_id = {row["candidate_id"]: row for row in kan_selected}
    profiles: set[str] = set()
    strict_count = 0
    spline_count = 0
    for factor_id in kan_ids:
        factor = _mapping(factors[factor_id], label=f"KAN factor {factor_id}")
        factor_lineage = _mapping(
            lineage.get(factor_id), label=f"KAN lineage {factor_id}"
        )
        score = score_by_id[factor_id]
        global_index = factor_lineage.get("global_attempt_index")
        run = _mapping(
            profile_by_index.get(global_index), label=f"KAN replay {factor_id}"
        )
        ast = AstNode.from_dict(
            _mapping(factor.get("ast"), label=f"KAN AST {factor_id}")
        )
        canonical = ast.identity
        if (
            factor.get("canonical_hash") != canonical
            or factor.get("global_attempt_index") != global_index
            or factor_lineage.get("canonical_hash") != canonical
            or score.get("canonical_hash") != canonical
            or score.get("attempt_index") != global_index
            or run.get("candidate_ast_sha256") != canonical
            or run.get("profile") != score.get("profile")
        ):
            raise ValueError(
                f"selected KAN lineage does not independently replay: {factor_id}"
            )
        profiles.add(str(score.get("profile")))
        strict_count += int(_strict_kan_ast(score, ast))
        spline_count += int(
            "spline" in json.dumps(ast.to_dict(), sort_keys=True).lower()
        )

    cards_path, cards = children["mechanism_cards"]
    blind_path, blind = children["blind_review_package"]
    if (
        cards.get("selected_factor_ids") != kan_ids
        or cards.get("card_count") != library_size
        or blind.get("blind_item_count") != library_size
        or cards.get("anonymous_mapping_sha256")
        != blind.get("anonymous_mapping_sha256")
    ):
        raise ValueError(
            "mechanism cards and blind package do not match selected KAN factors"
        )
    mapping_path = cards_path / "blind_anonymous_mapping.json"
    mapping_hash = sha256_file(mapping_path)
    if mapping_hash != cards.get("anonymous_mapping_sha256"):
        raise ValueError("mechanism-card anonymous mapping hash is invalid")
    anonymous_mapping = _read_json(mapping_path, label="blind anonymous mapping")
    expected_blind_ids = [f"B{index:03d}" for index in range(1, library_size + 1)]
    if (
        list(anonymous_mapping) != expected_blind_ids
        or list(anonymous_mapping.values()) != kan_ids
        or len(set(anonymous_mapping.values())) != library_size
    ):
        raise ValueError("blind anonymous mapping differs from selected KAN factors")
    card_rows = _json_lines(
        cards_path / "mechanism_cards.jsonl",
        label="mechanism cards",
        expected_count=library_size,
    )
    if [row.get("factor_id") for row in card_rows] != kan_ids:
        raise ValueError(
            "mechanism-card factor order differs from selected KAN factors"
        )
    for row in card_rows:
        factor_id = str(row["factor_id"])
        card = _mapping(row.get("card"), label=f"mechanism card {factor_id}")
        identity = _mapping(
            card.get("identity_and_canonical_ast"),
            label=f"mechanism identity {factor_id}",
        )
        if (
            identity.get("factor_id") != factor_id
            or identity.get("canonical_hash") != lineage[factor_id]["canonical_hash"]
        ):
            raise ValueError(f"mechanism-card lineage differs for {factor_id}")
    blind_record = _read_json(
        blind_path / "blind_review_package.json", label="blind package"
    )
    blind_items = blind_record.get("items")
    if (
        blind_record.get("review_status") != "pending_human_review"
        or not isinstance(blind_items, Mapping)
        or list(blind_items) != expected_blind_ids
    ):
        raise ValueError(
            "published blind package makes an unsupported human-review claim"
        )

    validation = _mapping(config.get("data"), label="protocol data").get("validation")
    if not isinstance(validation, list) or len(validation) != 2:
        raise ValueError("frozen validation period is invalid")
    rank_settings = _mapping(
        _mapping(config.get("diversity_metrics"), label="diversity metrics").get(
            "selected_library_effective_rank"
        ),
        label="effective-rank settings",
    )
    minimum_joint_rows = int(rank_settings.get("minimum_joint_rows", 2))
    kan_rank = _effective_rank(
        children["kan_library"][0] / "factor_panel.parquet",
        kan_ids,
        validation,
        minimum_joint_rows=minimum_joint_rows,
    )
    gp_rank = _effective_rank(
        children["gp_control_library"][0] / "factor_panel.parquet",
        library_ids["gp_control_library"],
        validation,
        minimum_joint_rows=minimum_joint_rows,
    )
    evidence = {
        "production_library_size": library_size,
        "production_profile_count": len(profiles),
        "production_strict_fraction": strict_count / library_size,
        "production_mechanism_card_fraction": len(card_rows) / library_size,
        "production_kan_mined_fraction": 1.0,
        "production_lineage_fraction": 1.0,
        "production_independent_replay_fraction": 1.0,
        "production_max_spline_ratio": spline_count / library_size,
        "gp_control_library_size": len(library_ids["gp_control_library"]),
        "blackbox_control_output_count": int(blackbox["control_count"]),
        "permutation_control_library_size": len(
            library_ids["permutation_control_library"]
        ),
        "permutation_false_positive_count": false_positive_count,
        "kan_unique_admitted_count": kan_admitted,
        "gp_unique_admitted_count": gp_admitted,
        "kan_selected_library_effective_rank": kan_rank,
        "gp_selected_library_effective_rank": gp_rank,
        "blind_review_package_sha256": sha256_file(blind_path / "manifest.json"),
        "human_blind_review": {"status": "pending", "reviews": []},
    }
    bindings = {
        "mining_manifest_sha256": sha256_file(top_path),
        "mining_child_manifest_sha256": dict(child_hashes),
        "blind_review_package_manifest_sha256": sha256_file(
            blind_path / "manifest.json"
        ),
    }
    return evidence, bindings


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_exclusive(path: Path, body: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(body)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("decision artifact write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def stage_v2_decision_artifact(
    workspace: Path | str, staging_path: Path | str
) -> StagedDecisionArtifact:
    """Derive and stage the decision; consume no final-publication authority."""
    context = _load_context(workspace)
    verify_implementation_lock(context["root"])
    _verify_development_claim(context)
    opening, opening_path = _verify_opening(context)
    evaluations, daily, calendar, evaluation_hashes = _load_evaluations(
        context, opening, opening_path
    )
    evidence, mining_bindings = _load_mining_evidence(context, opening)
    decision = decide_s2a_v2(evaluations, daily, evidence, context["config"], calendar)
    source_bindings = {
        "base_lock_sha256": sha256_file(context["lock_path"]),
        "config_sha256": context["config_sha256"],
        "implementation_lock_sha256": context["implementation_sha256"],
        "development_opening_sha256": sha256_file(opening_path),
        "development_topology_sha256": context["topology"].topology_sha256,
        "evaluation_manifest_sha256": evaluation_hashes,
        **mining_bindings,
    }
    payload = {
        "schema_version": "mirage_s2a_v2_decision_payload_v1",
        "decision": decision,
        "derived_evidence": evidence,
        "source_bindings": source_bindings,
    }
    raw = Path(staging_path)
    if not raw.name.endswith(".staging"):
        raise ValueError("decision staging path must end with .staging")
    parent = raw.parent.resolve(strict=True)
    if (
        not parent.is_dir()
        or parent.is_symlink()
        or not parent.is_relative_to(context["root"])
    ):
        raise ValueError("decision staging parent must be a real workspace directory")
    staging = parent / raw.name
    os.mkdir(staging, 0o700)
    try:
        decision_body = _canonical_bytes(payload)
        _write_exclusive(staging / "decision.json", decision_body)
        manifest = {
            "schema_version": "mirage_s2a_v2_decision_artifact_v1",
            "publication_state": "staged_unpublished",
            "role": "s2a_v2_decision",
            "protocol_id": context["config"].get("protocol_id"),
            "stage": "S2a",
            "outcome": decision["outcome"],
            "formal_promotion_allowed": False,
            "graph_unlock_allowed": decision["graph_unlock_allowed"],
            "final_decision_authority_consumed": False,
            "final_decision_authority_required_at_publication": True,
            "development_topology_sha256": context["topology"].topology_sha256,
            "evaluation_manifest_sha256": evaluation_hashes,
            "source_bindings": source_bindings,
            "files": {"decision.json": hashlib.sha256(decision_body).hexdigest()},
        }
        _write_exclusive(staging / "manifest.json", _canonical_bytes(manifest))
    except BaseException:
        for child in staging.iterdir():
            child.unlink()
        staging.rmdir()
        raise
    manifest_path = staging / "manifest.json"
    return StagedDecisionArtifact(
        path=staging,
        manifest=manifest,
        decision=decision,
        evidence=evidence,
        manifest_sha256=sha256_file(manifest_path),
    )


__all__ = ["StagedDecisionArtifact", "stage_v2_decision_artifact"]
