"""Fail-closed reuse of a complete mining topology across protocol identities."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import yaml

from mirage_kan.data.pit import sha256_file


SCHEMA_VERSION = "mirage_mining_rebind_receipt_v1"
_MINING_KEYS = (
    "mining_run",
    "kan_library",
    "gp_control_library",
    "permutation_control_library",
    "blackbox_control",
    "mechanism_cards",
    "blind_review_package",
)
_CHILD_KEYS = frozenset(_MINING_KEYS[1:])


def _authority_receipt_sha256(payload: object) -> str:
    body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not a mapping")
    return value


def _read_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _workspace_file(root: Path, value: object, *, label: str) -> Path:
    if isinstance(value, Path):
        raw = value
    elif isinstance(value, str) and value:
        raw = Path(value)
    else:
        raise ValueError(f"{label} path is missing")
    path = raw if raw.is_absolute() else root / raw
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise ValueError(f"{label} path escapes the workspace or is missing") from error
    if path.is_symlink() or not resolved.is_file():
        raise ValueError(f"{label} is not a regular non-symlink file")
    return resolved


def _workspace_directory(root: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path is missing")
    raw = Path(value)
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"{label} path is not workspace-relative")
    path = root / raw
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise ValueError(f"{label} path escapes the workspace or is missing") from error
    if path.is_symlink() or not resolved.is_dir():
        raise ValueError(f"{label} is not a real directory")
    return resolved


def _locked_config(
    root: Path, lock_path: Path, *, label: str
) -> tuple[dict[str, object], Path, dict[str, object]]:
    lock = _read_json(lock_path, label=f"{label} base lock")
    protocol = _mapping(lock.get("protocol"), label=f"{label} protocol pin")
    protocol_id = protocol.get("protocol_id")
    if not isinstance(protocol_id, str) or not protocol_id:
        raise ValueError(f"{label} protocol ID is invalid")
    config_path = _workspace_file(
        root, protocol.get("path"), label=f"{label} protocol config"
    )
    if sha256_file(config_path) != protocol.get("sha256"):
        raise ValueError(f"{label} protocol config hash mismatch")
    try:
        config_value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read {label} protocol config") from error
    config = dict(_mapping(config_value, label=f"{label} protocol config"))
    if config.get("protocol_id") != protocol_id:
        raise ValueError(f"{label} config has the wrong protocol ID")
    return lock, config_path, config


def _scientific_payload(config: Mapping[str, object]) -> dict[str, object]:
    payload = copy.deepcopy(dict(config))
    payload.pop("protocol_id", None)
    payload.pop("artifact_paths", None)
    payload.pop("mining_source", None)
    claim = payload.get("claim_boundary")
    if isinstance(claim, dict):
        claim.pop("graph_unlock_allowed", None)
    return payload


def _artifact_identity(
    root: Path,
    path_value: object,
    *,
    key: str,
    topology_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    path = _workspace_directory(root, path_value, label=f"source artifact {key}")
    manifest_path = path / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"source artifact manifest is invalid: {key}")
    manifest = _read_json(manifest_path, label=f"source artifact manifest {key}")
    if (
        manifest.get("topology_key") != key
        or manifest.get("topology_sha256") != topology_sha256
    ):
        raise ValueError(f"source artifact topology identity mismatch: {key}")
    declared = _mapping(
        manifest.get("files"), label=f"source artifact declared files {key}"
    )
    entries = tuple(path.iterdir())
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ValueError(f"source artifact is not a flat regular-file bundle: {key}")
    if {entry.name for entry in entries} != set(declared) | {"manifest.json"}:
        raise ValueError(f"source artifact file set differs from manifest: {key}")
    files: dict[str, dict[str, object]] = {
        "manifest.json": {
            "sha256": sha256_file(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
        }
    }
    for filename, expected in declared.items():
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"source artifact has an unsafe filename: {key}")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"source artifact has an invalid file hash: {key}")
        observed = sha256_file(path / filename)
        if observed != expected:
            raise ValueError(f"source artifact file hash mismatch: {key}/{filename}")
        files[filename] = {
            "sha256": observed,
            "size_bytes": (path / filename).stat().st_size,
        }
    return (
        {
            "path": path.relative_to(root).as_posix(),
            "manifest_sha256": files["manifest.json"]["sha256"],
            "files": dict(sorted(files.items())),
        },
        manifest,
    )


def build_mining_rebind_receipt(
    workspace: Path | str,
    *,
    target_base_lock_path: Path | str,
    target_implementation_lock_path: Path | str,
) -> dict[str, object]:
    """Build a deterministic receipt without reading labels or copying payloads."""
    root = Path(workspace).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("rebind workspace is not a directory")
    target_base = _workspace_file(
        root, target_base_lock_path, label="target base lock"
    )
    target_implementation = _workspace_file(
        root, target_implementation_lock_path, label="target implementation lock"
    )
    _, target_config_path, target_config = _locked_config(
        root, target_base, label="target"
    )
    target_protocol = target_config["protocol_id"]
    target_implementation_record = _read_json(
        target_implementation, label="target implementation lock"
    )
    if target_implementation_record.get("protocol_id") != target_protocol:
        raise ValueError("target implementation lock has the wrong protocol")

    mining_source = _mapping(
        target_config.get("mining_source"), label="target mining source"
    )
    if mining_source.get("mode") != "verified_cross_protocol_rebind":
        raise ValueError("target mining source does not select verified rebind")
    absence = _mapping(
        mining_source.get("absence_contract"), label="rebind absence contract"
    )
    expected_absence = {
        "source_development_preclaim",
        "source_development_opening",
        "target_mining_preclaim",
        "target_mining_entitlement",
    }
    if set(absence) != expected_absence:
        raise ValueError("rebind absence contract is incomplete")
    for label, raw_path in absence.items():
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"forbidden opening path is invalid: {label}")
        path = Path(raw_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"forbidden opening path escapes workspace: {label}")
        candidate = root / path
        if candidate.exists() or candidate.is_symlink():
            raise ValueError(f"forbidden opening exists: {label}")
    source_base = _workspace_file(
        root, mining_source.get("source_base_lock"), label="source base lock"
    )
    source_implementation = _workspace_file(
        root,
        mining_source.get("source_implementation_lock"),
        label="source implementation lock",
    )
    _, source_config_path, source_config = _locked_config(
        root, source_base, label="source"
    )
    source_protocol = source_config["protocol_id"]
    if (
        mining_source.get("source_protocol_id") != source_protocol
        or source_protocol == target_protocol
    ):
        raise ValueError("source and target protocol identities are invalid")
    source_implementation_record = _read_json(
        source_implementation, label="source implementation lock"
    )
    if source_implementation_record.get("protocol_id") != source_protocol:
        raise ValueError("source implementation lock has the wrong protocol")
    if _scientific_payload(source_config) != _scientific_payload(target_config):
        raise ValueError("source and target scientific configuration differs")

    source_paths = _mapping(
        mining_source.get("source_artifact_paths"),
        label="source artifact paths",
    )
    if set(source_paths) != set(_MINING_KEYS):
        raise ValueError("source artifact paths do not name the full mining topology")
    source_config_paths = _mapping(
        source_config.get("artifact_paths"), label="source configured artifact paths"
    )
    if any(source_paths[key] != source_config_paths.get(key) for key in _MINING_KEYS):
        raise ValueError("source artifact paths differ from the source protocol")

    entitlement_path = _workspace_file(
        root,
        mining_source.get("source_mining_entitlement"),
        label="source mining entitlement",
    )
    preclaim_path = _workspace_file(
        root,
        mining_source.get("source_mining_preclaim"),
        label="source mining preclaim",
    )
    entitlement = _read_json(entitlement_path, label="source mining entitlement")
    preclaim = _read_json(preclaim_path, label="source mining preclaim")
    topology_sha256 = entitlement.get("topology_sha256")
    expected_entitlement = {
        "schema_version": "mirage_mining_entitlement_v2",
        "protocol_id": source_protocol,
        "state": "consumed_before_first_label_access",
        "base_lock_sha256": sha256_file(source_base),
        "config_sha256": sha256_file(source_config_path),
        "implementation_lock_sha256": sha256_file(source_implementation),
        "topology_preclaim_sha256": sha256_file(preclaim_path),
    }
    if (
        not isinstance(topology_sha256, str)
        or len(topology_sha256) != 64
        or any(entitlement.get(key) != value for key, value in expected_entitlement.items())
        or preclaim.get("protocol_id") != source_protocol
        or preclaim.get("topology_sha256") != topology_sha256
    ):
        raise ValueError("source mining entitlement identity mismatch")
    authority_receipt_sha256 = entitlement.get("authority_receipt_sha256")
    if not isinstance(authority_receipt_sha256, str):
        raise ValueError("source mining entitlement lacks its authority receipt")
    receipt_directory = (
        root / "governance" / "authority" / source_protocol / "receipts"
    )
    if receipt_directory.is_symlink() or not receipt_directory.is_dir():
        raise ValueError("source mining authority receipt directory is invalid")
    authority_matches: list[tuple[Path, dict[str, object]]] = []
    for candidate in sorted(receipt_directory.glob("*.json")):
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError("source mining authority receipt is not a regular file")
        record = _read_json(candidate, label="source mining authority receipt")
        if record.get("receipt_sha256") == authority_receipt_sha256:
            authority_matches.append((candidate, record))
    if len(authority_matches) != 1:
        raise ValueError("source mining authority receipt identity is not unique")
    authority_path, authority_record = authority_matches[0]
    authority_payload = {
        key: value
        for key, value in authority_record.items()
        if key not in {"receipt_sha256", "capability"}
    }
    computed_authority_sha256 = _authority_receipt_sha256(authority_payload)
    if (
        authority_record.get("schema_version") != "mirage_authority_receipt_v2"
        or authority_record.get("protocol_id") != source_protocol
        or authority_record.get("sequence") != 1
        or authority_record.get("boundary") != "before_first_label_access"
        or authority_record.get("arm") is not None
        or authority_record.get("base_lock_sha256") != sha256_file(source_base)
        or computed_authority_sha256 != authority_receipt_sha256
    ):
        raise ValueError("source mining authority receipt identity mismatch")
    attempt_budget = entitlement.get("attempt_budget")
    expected_budget = _mapping(
        source_config.get("kan_e3"), label="source KAN configuration"
    ).get("total_miner_attempts")
    if attempt_budget != expected_budget:
        raise ValueError("source mining entitlement attempt budget mismatch")

    artifacts: dict[str, dict[str, object]] = {}
    manifests: dict[str, dict[str, object]] = {}
    for key in _MINING_KEYS:
        artifacts[key], manifests[key] = _artifact_identity(
            root,
            source_paths[key],
            key=key,
            topology_sha256=topology_sha256,
        )
    file_count = sum(len(record["files"]) for record in artifacts.values())
    total_bytes = sum(
        int(file_record["size_bytes"])
        for artifact in artifacts.values()
        for file_record in artifact["files"].values()
    )
    inventory_sha256 = hashlib.sha256(
        json.dumps(artifacts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    top = manifests["mining_run"]
    if (
        top.get("schema_version") != "mirage_s2a_v2_staging_bundle_v1"
        or top.get("role") != "mining_top_bundle"
        or top.get("published_child_topology_sha256") != topology_sha256
    ):
        raise ValueError("source mining top manifest identity mismatch")
    expected_children = {
        key: artifacts[key]["manifest_sha256"] for key in _MINING_KEYS[1:]
    }
    if (
        top.get("child_manifests") != expected_children
        or top.get("child_manifest_sha256") != expected_children
    ):
        raise ValueError("source mining top does not bind every child manifest")
    published_paths = _mapping(
        top.get("published_child_paths"), label="source published child paths"
    )
    if set(published_paths) != _CHILD_KEYS:
        raise ValueError("source mining top child path set is incomplete")
    for key in _MINING_KEYS[1:]:
        recorded = Path(str(published_paths[key])).resolve(strict=True)
        expected = (root / str(source_paths[key])).resolve(strict=True)
        if recorded != expected:
            raise ValueError(f"source mining top child path mismatch: {key}")

    return {
        "schema_version": SCHEMA_VERSION,
        "state": "verified_without_label_access",
        "source": {
            "protocol_id": source_protocol,
            "base_lock": {
                "path": source_base.relative_to(root).as_posix(),
                "sha256": sha256_file(source_base),
            },
            "config": {
                "path": source_config_path.relative_to(root).as_posix(),
                "sha256": sha256_file(source_config_path),
            },
            "implementation_lock": {
                "path": source_implementation.relative_to(root).as_posix(),
                "sha256": sha256_file(source_implementation),
            },
            "preclaim": {
                "path": preclaim_path.relative_to(root).as_posix(),
                "sha256": sha256_file(preclaim_path),
            },
            "entitlement": {
                "path": entitlement_path.relative_to(root).as_posix(),
                "sha256": sha256_file(entitlement_path),
            },
            "authority_receipt": {
                "path": authority_path.relative_to(root).as_posix(),
                "sha256": sha256_file(authority_path),
                "receipt_sha256": authority_receipt_sha256,
            },
            "topology_sha256": topology_sha256,
            "artifacts": artifacts,
            "inventory": {
                "file_count": file_count,
                "total_bytes": total_bytes,
                "sha256": inventory_sha256,
            },
        },
        "target": {
            "protocol_id": target_protocol,
            "base_lock": {
                "path": target_base.relative_to(root).as_posix(),
                "sha256": sha256_file(target_base),
            },
            "config": {
                "path": target_config_path.relative_to(root).as_posix(),
                "sha256": sha256_file(target_config_path),
            },
            "implementation_lock": {
                "path": target_implementation.relative_to(root).as_posix(),
                "sha256": sha256_file(target_implementation),
            },
        },
        "contract": {
            "label_access_performed": False,
            "reselection_performed": False,
            "reordering_performed": False,
            "retuning_performed": False,
            "source_payload_copy_performed": False,
        },
    }


def verify_mining_rebind_receipt(
    workspace: Path | str,
    *,
    target_base_lock_path: Path | str,
    target_implementation_lock_path: Path | str,
) -> dict[str, object]:
    """Verify the exclusive receipt and rebuild every live source identity."""
    root = Path(workspace).resolve(strict=True)
    target_base = _workspace_file(
        root, target_base_lock_path, label="target base lock"
    )
    _, _, target_config = _locked_config(root, target_base, label="target")
    mining_source = _mapping(
        target_config.get("mining_source"), label="target mining source"
    )
    receipt_path = _workspace_file(
        root, mining_source.get("rebind_receipt"), label="mining rebind receipt"
    )
    observed = _read_json(receipt_path, label="mining rebind receipt")
    try:
        expected = build_mining_rebind_receipt(
            root,
            target_base_lock_path=target_base,
            target_implementation_lock_path=target_implementation_lock_path,
        )
    except ValueError as error:
        raise ValueError(
            "source artifact changed or rebind contract is no longer valid"
        ) from error
    if observed != expected:
        raise ValueError("mining rebind receipt differs from live verified identities")
    return observed


def write_mining_rebind_receipt(
    workspace: Path | str,
    *,
    target_base_lock_path: Path | str,
    target_implementation_lock_path: Path | str,
) -> dict[str, object]:
    """Build and atomically consume the configured receipt path exactly once."""
    root = Path(workspace).resolve(strict=True)
    target_base = _workspace_file(
        root, target_base_lock_path, label="target base lock"
    )
    _, _, target_config = _locked_config(root, target_base, label="target")
    mining_source = _mapping(
        target_config.get("mining_source"), label="target mining source"
    )
    raw_path = mining_source.get("rebind_receipt")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("mining rebind receipt path is missing")
    path = root / raw_path
    if path.parent.resolve(strict=True) != root / "governance/openings":
        raise ValueError("mining rebind receipt path escapes governance/openings")
    payload = build_mining_rebind_receipt(
        root,
        target_base_lock_path=target_base,
        target_implementation_lock_path=target_implementation_lock_path,
    )
    body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return payload


__all__ = [
    "SCHEMA_VERSION",
    "build_mining_rebind_receipt",
    "verify_mining_rebind_receipt",
    "write_mining_rebind_receipt",
]
