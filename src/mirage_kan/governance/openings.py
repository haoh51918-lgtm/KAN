"""Single-use label entitlement and development opening for active S2a."""

from __future__ import annotations

import json
import hashlib
import os
from datetime import date
from pathlib import Path
from typing import Mapping

import yaml
import pandas as pd

from mirage_kan.artifacts.topology import TopologyTransaction
from mirage_kan.data.pit import sha256_file
from mirage_kan.governance.authority import AuthorityGuard
from mirage_kan.governance.mining_rebind import verify_mining_rebind_receipt
from mirage_kan.protocol import BASE_LOCK


def _load_context(workspace: Path | str) -> dict[str, object]:
    root = Path(workspace).resolve(strict=True)
    lock_path = root / BASE_LOCK
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    protocol = lock["protocol"]
    config_path = root / protocol["path"]
    if sha256_file(config_path) != protocol["sha256"]:
        raise ValueError("opening config hash differs from the base lock")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("opening config has the wrong protocol")
    implementation_path = root / config["artifact_paths"]["implementation_lock"]
    if not implementation_path.is_file() or implementation_path.is_symlink():
        raise ValueError("opening implementation lock is missing or invalid")
    return {
        "workspace": root,
        "base_lock": lock,
        "base_lock_path": lock_path,
        "base_lock_sha256": sha256_file(lock_path),
        "config": config,
        "config_path": config_path,
        "config_sha256": sha256_file(config_path),
        "implementation_path": implementation_path,
        "implementation_sha256": sha256_file(implementation_path),
    }


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not a mapping")
    return value


def _control_path(context: Mapping[str, object], key: str) -> Path:
    root = context["workspace"]
    raw = Path(context["config"]["artifact_paths"][key])
    if (
        raw.is_absolute()
        or ".." in raw.parts
        or len(raw.parts) != 3
        or raw.parts[:2] != ("governance", "openings")
        or raw.suffix != ".json"
    ):
        raise ValueError(f"{key} is not a fixed governance opening JSON path")
    path = root / raw
    if path.parent.resolve(strict=True) != root / "governance/openings":
        raise ValueError(f"{key} path escapes its frozen parent")
    if path.is_symlink():
        raise ValueError(f"{key} path is a symlink")
    return path


def _canonical_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_exclusive(path: Path, payload: object) -> None:
    body = _canonical_bytes(payload)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        view = memoryview(body)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("opening receipt write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def _period_iso_strings(value: object, *, field: str) -> tuple[str, str]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field} must contain exactly two dates")
    normalized: list[str] = []
    for scalar in value:
        if type(scalar) is date:
            normalized.append(scalar.isoformat())
            continue
        if type(scalar) is not str:
            raise ValueError(f"{field} scalar must be a date or ISO date string")
        try:
            normalized.append(date.fromisoformat(scalar).isoformat())
        except ValueError as error:
            raise ValueError(
                f"{field} scalar must be a date or ISO date string"
            ) from error
    return normalized[0], normalized[1]


def _verify_claims(topology: TopologyTransaction) -> None:
    for key, path in topology.targets.items():
        marker = path / ".INCOMPLETE"
        if not marker.is_file() or marker.is_symlink():
            raise ValueError(f"mining entitlement target is not claimed: {key}")
        record = json.loads(marker.read_text(encoding="utf-8"))
        if (
            record.get("topology_sha256") != topology.topology_sha256
            or record.get("topology_key") != key
        ):
            raise ValueError(f"mining entitlement target claim is not owned: {key}")


def consume_mining_entitlement(
    workspace: Path | str,
    topology: TopologyTransaction,
    authority_guard: AuthorityGuard,
    authority_capability: str,
) -> dict[str, object]:
    """Consume the active label/search entitlement before labels are loaded."""
    from mirage_kan.governance.implementation_lock import verify_implementation_lock

    if topology.phase != "mining":
        raise ValueError("mining entitlement requires the mining topology")
    context = _load_context(workspace)
    verify_implementation_lock(context["workspace"])
    if topology.workspace != context["workspace"]:
        raise ValueError("mining entitlement topology belongs to another workspace")
    _verify_claims(topology)
    authority = authority_guard.verify_capability(
        authority_capability, boundary="before_first_label_access"
    )
    config = context["config"]
    train = _period_iso_strings(config["data"]["train"], field="train")
    validation = _period_iso_strings(
        config["data"]["validation"], field="validation"
    )
    record = {
        "schema_version": "mirage_mining_entitlement_v2",
        "protocol_id": config["protocol_id"],
        "state": "consumed_before_first_label_access",
        "topology_sha256": topology.topology_sha256,
        "topology_preclaim_sha256": sha256_file(topology.preclaim_path),
        "authority_receipt_sha256": authority.receipt_sha256,
        "base_lock_sha256": context["base_lock_sha256"],
        "config_sha256": context["config_sha256"],
        "implementation_lock_sha256": context["implementation_sha256"],
        "data_identity": context["base_lock"].get("data"),
        "train": list(train),
        "validation": list(validation),
        "attempt_budget": int(config["kan_e3"]["total_miner_attempts"]),
    }
    _write_exclusive(_control_path(context, "mining_entitlement"), record)
    return record


def _published_mining_manifests(
    topology: TopologyTransaction,
) -> dict[str, str]:
    manifests: dict[str, str] = {}
    records: dict[str, Mapping[str, object]] = {}
    for key, path in topology.targets.items():
        if (path / ".INCOMPLETE").exists() or (path / "terminal_failure.json").exists():
            raise ValueError(f"mining manifest is not immutable and successful: {key}")
        manifest = path / "manifest.json"
        if not manifest.is_file() or manifest.is_symlink():
            raise ValueError(f"mining manifest is missing: {key}")
        record = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(record, Mapping):
            raise ValueError(f"mining manifest is not a mapping: {key}")
        if (
            record.get("topology_sha256") != topology.topology_sha256
            or record.get("topology_key") != key
        ):
            raise ValueError(f"mining manifest ownership differs: {key}")
        manifests[key] = sha256_file(manifest)
        records[key] = record
    top = records[topology.top_key].get("child_manifests")
    expected_children = {key: manifests[key] for key in topology.child_keys}
    if top != expected_children:
        raise ValueError("mining top manifest does not bind final child manifests")
    return manifests


def _verified_mining_entitlement(
    context: Mapping[str, object], topology_sha256: str
) -> tuple[dict[str, object], str]:
    path = _control_path(context, "mining_entitlement")
    if not path.is_file() or path.is_symlink():
        raise ValueError("development opening requires the mining entitlement")
    record = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": "mirage_mining_entitlement_v2",
        "protocol_id": context["config"]["protocol_id"],
        "state": "consumed_before_first_label_access",
        "topology_sha256": topology_sha256,
        "base_lock_sha256": context["base_lock_sha256"],
        "config_sha256": context["config_sha256"],
        "implementation_lock_sha256": context["implementation_sha256"],
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"mining entitlement identity mismatch: {key}")
    return record, sha256_file(path)


def _provider_identity(context: Mapping[str, object]) -> dict[str, object]:
    implementation = json.loads(
        context["implementation_path"].read_text(encoding="utf-8")
    )
    provider = implementation.get("qlib_provider")
    fields = (
        "path",
        "tree_sha256",
        "stat_inventory_sha256",
        "file_count",
        "total_bytes",
    )
    if not isinstance(provider, Mapping) or set(fields).difference(provider):
        raise ValueError("implementation lock lacks the QLib provider identity")
    result = {field: provider[field] for field in fields}
    if (
        not isinstance(result["path"], str)
        or not isinstance(result["tree_sha256"], str)
        or len(result["tree_sha256"]) != 64
        or not isinstance(result["stat_inventory_sha256"], str)
        or len(result["stat_inventory_sha256"]) != 64
        or type(result["file_count"]) is not int
        or type(result["total_bytes"]) is not int
    ):
        raise ValueError("implementation lock QLib provider identity is invalid")
    return result


def _development_calendar(
    context: Mapping[str, object],
) -> tuple[pd.Timestamp, ...]:
    data = context["base_lock"].get("data")
    if not isinstance(data, Mapping):
        raise ValueError("base lock lacks the PIT cache identity")
    cache_path = Path(str(data.get("cache_path"))).resolve(strict=True)
    if not cache_path.is_file() or cache_path.is_symlink():
        raise ValueError("development calendar PIT cache is invalid")
    if sha256_file(cache_path) != data.get("cache_sha256"):
        raise ValueError("development calendar PIT cache hash mismatch")
    period = _period_iso_strings(
        context["config"]["data"]["development_test"], field="development_test"
    )
    start, end = (pd.Timestamp(value) for value in period)
    frame = pd.read_parquet(
        cache_path,
        columns=["datetime"],
        filters=[("datetime", ">=", start), ("datetime", "<=", end)],
    )
    dates = tuple(
        pd.DatetimeIndex(pd.to_datetime(frame["datetime"]).unique()).sort_values()
    )
    if not dates or dates[0] < start or dates[-1] > end:
        raise ValueError("development calendar is empty or outside the frozen period")
    return tuple(pd.Timestamp(value) for value in dates)


def _calendar_sha256(dates: tuple[pd.Timestamp, ...]) -> str:
    payload = [value.isoformat() for value in dates]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def consume_development_opening(
    workspace: Path | str,
    mining_topology: TopologyTransaction,
    development_topology: TopologyTransaction,
    authority_guard: AuthorityGuard,
    authority_capability: str,
) -> dict[str, object]:
    """Open 2022--2025 once, only after immutable mining/control publication."""
    from mirage_kan.governance.implementation_lock import verify_implementation_lock

    if mining_topology.phase != "mining" or development_topology.phase != "development":
        raise ValueError("development opening received the wrong topology phases")
    context = _load_context(workspace)
    verify_implementation_lock(context["workspace"])
    if (
        mining_topology.workspace != context["workspace"]
        or development_topology.workspace != context["workspace"]
    ):
        raise ValueError("development opening topology belongs to another workspace")
    preclaim = json.loads(
        development_topology.preclaim_path.read_text(encoding="utf-8")
    )
    if preclaim.get("topology_sha256") != development_topology.topology_sha256:
        raise ValueError("development topology was not prospectively preclaimed")
    _, entitlement_sha256 = _verified_mining_entitlement(
        context, mining_topology.topology_sha256
    )
    authority = authority_guard.verify_capability(
        authority_capability, boundary="before_development_opening"
    )
    manifests = _published_mining_manifests(mining_topology)
    dates = _development_calendar(context)
    provider = _provider_identity(context)
    config = context["config"]
    evaluations = config["artifact_paths"]["evaluations"]
    development_period = _period_iso_strings(
        config["data"]["development_test"], field="development_test"
    )
    record = {
        "schema_version": "mirage_s2a_development_opening_v2",
        "protocol_id": config["protocol_id"],
        "state": "consumed_before_first_development_access",
        "authority_receipt": {
            "receipt_sha256": authority.receipt_sha256,
            "sequence": authority.sequence,
            "boundary": authority.boundary,
            "authority_sha256": authority.authority_sha256,
            "base_lock_sha256": authority.base_lock_sha256,
            "capability_sha256": hashlib.sha256(
                authority_capability.encode("utf-8")
            ).hexdigest(),
        },
        "identity_pins": {
            "base_lock_sha256": context["base_lock_sha256"],
            "implementation_lock_sha256": context["implementation_sha256"],
            "mining_manifest_sha256": manifests[mining_topology.top_key],
            "provider_identity": provider,
        },
        "mining_entitlement_sha256": entitlement_sha256,
        "mining_topology_sha256": mining_topology.topology_sha256,
        "topology_sha256": development_topology.topology_sha256,
        "development_preclaim_sha256": sha256_file(development_topology.preclaim_path),
        "development_period": list(development_period),
        "development_calendar_sha256": _calendar_sha256(dates),
        "development_calendar_count": len(dates),
        "development_calendar_start": dates[0].isoformat(),
        "development_calendar_end": dates[-1].isoformat(),
        "evaluation_paths": dict(evaluations),
        "data_identity": context["base_lock"].get("data"),
        "baseline_identity": context["base_lock"].get("baseline_metric"),
        "mining_manifests": manifests,
    }
    path = _control_path(context, "development_opening")
    _write_exclusive(path, record)
    return record


def _verified_mining_rebind(
    context: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    mining_source = _mapping(
        context["config"].get("mining_source"), label="mining source"
    )
    raw_path = mining_source.get("rebind_receipt")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("mining rebind receipt path is missing")
    relative = Path(raw_path)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.parts[:2] != ("governance", "openings")
        or len(relative.parts) != 3
        or relative.suffix != ".json"
    ):
        raise ValueError("mining rebind receipt path is invalid")
    path = context["workspace"] / relative
    binding = verify_mining_rebind_receipt(
        context["workspace"],
        target_base_lock_path=context["base_lock_path"],
        target_implementation_lock_path=context["implementation_path"],
    )
    return binding, sha256_file(path)


def consume_rebound_development_opening(
    workspace: Path | str,
    development_topology: TopologyTransaction,
    authority_guard: AuthorityGuard,
    authority_capability: str,
) -> dict[str, object]:
    """Open development using a verified predecessor mining topology."""
    from mirage_kan.governance.implementation_lock import verify_implementation_lock

    if development_topology.phase != "development":
        raise ValueError("rebound development opening received the wrong topology")
    context = _load_context(workspace)
    verify_implementation_lock(context["workspace"])
    if development_topology.workspace != context["workspace"]:
        raise ValueError("development opening topology belongs to another workspace")
    preclaim = json.loads(
        development_topology.preclaim_path.read_text(encoding="utf-8")
    )
    if preclaim.get("topology_sha256") != development_topology.topology_sha256:
        raise ValueError("development topology was not prospectively preclaimed")
    binding, rebind_sha256 = _verified_mining_rebind(context)
    authority = authority_guard.verify_capability(
        authority_capability, boundary="before_development_opening"
    )
    dates = _development_calendar(context)
    provider = _provider_identity(context)
    config = context["config"]
    evaluations = config["artifact_paths"]["evaluations"]
    development_period = _period_iso_strings(
        config["data"]["development_test"], field="development_test"
    )
    source = _mapping(binding.get("source"), label="rebind source")
    artifacts = _mapping(source.get("artifacts"), label="rebind source artifacts")
    manifests = {
        key: _mapping(value, label=f"rebind artifact {key}")["manifest_sha256"]
        for key, value in artifacts.items()
    }
    top_manifest = manifests.get("mining_run")
    if not isinstance(top_manifest, str):
        raise ValueError("rebind source lacks the mining top manifest")
    record = {
        "schema_version": "mirage_s2a_development_opening_v3",
        "protocol_id": config["protocol_id"],
        "state": "consumed_before_first_development_access",
        "mining_authorization_kind": "verified_cross_protocol_rebind",
        "authority_receipt": {
            "receipt_sha256": authority.receipt_sha256,
            "sequence": authority.sequence,
            "boundary": authority.boundary,
            "authority_sha256": authority.authority_sha256,
            "base_lock_sha256": authority.base_lock_sha256,
            "capability_sha256": hashlib.sha256(
                authority_capability.encode("utf-8")
            ).hexdigest(),
        },
        "identity_pins": {
            "base_lock_sha256": context["base_lock_sha256"],
            "implementation_lock_sha256": context["implementation_sha256"],
            "mining_manifest_sha256": top_manifest,
            "mining_rebind_receipt_sha256": rebind_sha256,
            "provider_identity": provider,
        },
        "source_protocol_id": source.get("protocol_id"),
        "source_mining_entitlement_sha256": _mapping(
            source.get("entitlement"), label="source entitlement"
        ).get("sha256"),
        "source_mining_topology_sha256": source.get("topology_sha256"),
        "source_mining_manifests": manifests,
        "topology_sha256": development_topology.topology_sha256,
        "development_preclaim_sha256": sha256_file(
            development_topology.preclaim_path
        ),
        "development_period": list(development_period),
        "development_calendar_sha256": _calendar_sha256(dates),
        "development_calendar_count": len(dates),
        "development_calendar_start": dates[0].isoformat(),
        "development_calendar_end": dates[-1].isoformat(),
        "evaluation_paths": dict(evaluations),
        "data_identity": context["base_lock"].get("data"),
        "baseline_identity": context["base_lock"].get("baseline_metric"),
    }
    path = _control_path(context, "development_opening")
    _write_exclusive(path, record)
    return record


def verify_development_opening(workspace: Path | str) -> dict[str, object]:
    """Recheck the live lock chain and every immutable mining manifest."""
    context = _load_context(workspace)
    path = _control_path(context, "development_opening")
    if not path.is_file() or path.is_symlink():
        raise ValueError("development opening is missing")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema_version") == "mirage_s2a_development_opening_v3":
        return _verify_rebound_development_opening(context, record)
    pins = record.get("identity_pins")
    if not isinstance(pins, Mapping):
        raise ValueError("development opening lacks identity pins")
    expected_pins = {
        "base_lock_sha256": context["base_lock_sha256"],
        "implementation_lock_sha256": context["implementation_sha256"],
        "mining_manifest_sha256": record["mining_manifests"]["mining_run"],
        "provider_identity": _provider_identity(context),
    }
    if pins != expected_pins:
        raise ValueError("development opening live identity pins differ")
    _, entitlement_sha256 = _verified_mining_entitlement(
        context, record["mining_topology_sha256"]
    )
    if record.get("mining_entitlement_sha256") != entitlement_sha256:
        raise ValueError("development opening mining entitlement changed")
    dates = _development_calendar(context)
    if (
        record.get("development_calendar_sha256") != _calendar_sha256(dates)
        or record.get("development_calendar_count") != len(dates)
        or record.get("development_calendar_start") != dates[0].isoformat()
        or record.get("development_calendar_end") != dates[-1].isoformat()
    ):
        raise ValueError("development opening calendar changed")
    if (
        record.get("evaluation_paths")
        != context["config"]["artifact_paths"]["evaluations"]
    ):
        raise ValueError("development opening evaluation paths changed")
    development_period = _period_iso_strings(
        context["config"]["data"]["development_test"], field="development_test"
    )
    if record.get("development_period") != list(development_period):
        raise ValueError("development opening period changed")
    for key, expected in record.get("mining_manifests", {}).items():
        config = context["config"]["artifact_paths"]
        raw = config["mining_run"] if key == "mining_run" else config[key]
        manifest = context["workspace"] / raw / "manifest.json"
        if sha256_file(manifest) != expected:
            raise ValueError(f"development opening mining manifest changed: {key}")
    return record


def _verify_rebound_development_opening(
    context: Mapping[str, object], record: dict[str, object]
) -> dict[str, object]:
    if (
        record.get("protocol_id") != context["config"].get("protocol_id")
        or record.get("state") != "consumed_before_first_development_access"
        or record.get("mining_authorization_kind")
        != "verified_cross_protocol_rebind"
    ):
        raise ValueError("rebound development opening has an invalid identity")
    binding, rebind_sha256 = _verified_mining_rebind(context)
    source = _mapping(binding.get("source"), label="rebind source")
    artifacts = _mapping(source.get("artifacts"), label="rebind source artifacts")
    manifests = {
        key: _mapping(value, label=f"rebind artifact {key}")["manifest_sha256"]
        for key, value in artifacts.items()
    }
    pins = _mapping(record.get("identity_pins"), label="development identity pins")
    expected_pins = {
        "base_lock_sha256": context["base_lock_sha256"],
        "implementation_lock_sha256": context["implementation_sha256"],
        "mining_manifest_sha256": manifests["mining_run"],
        "mining_rebind_receipt_sha256": rebind_sha256,
        "provider_identity": _provider_identity(context),
    }
    if dict(pins) != expected_pins:
        raise ValueError("rebound development opening live identity pins differ")
    if (
        record.get("source_protocol_id") != source.get("protocol_id")
        or record.get("source_mining_entitlement_sha256")
        != _mapping(source.get("entitlement"), label="source entitlement").get(
            "sha256"
        )
        or record.get("source_mining_topology_sha256")
        != source.get("topology_sha256")
        or record.get("source_mining_manifests") != manifests
    ):
        raise ValueError("rebound development opening source identity changed")
    dates = _development_calendar(context)
    if (
        record.get("development_calendar_sha256") != _calendar_sha256(dates)
        or record.get("development_calendar_count") != len(dates)
        or record.get("development_calendar_start") != dates[0].isoformat()
        or record.get("development_calendar_end") != dates[-1].isoformat()
    ):
        raise ValueError("rebound development opening calendar changed")
    if record.get("evaluation_paths") != context["config"]["artifact_paths"].get(
        "evaluations"
    ):
        raise ValueError("rebound development opening evaluation paths changed")
    development_period = _period_iso_strings(
        context["config"]["data"]["development_test"], field="development_test"
    )
    if record.get("development_period") != list(development_period):
        raise ValueError("rebound development opening period changed")
    return record


__all__ = [
    "consume_development_opening",
    "consume_mining_entitlement",
    "consume_rebound_development_opening",
    "verify_development_opening",
]
