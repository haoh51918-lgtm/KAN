"""No-replace multi-artifact topology transactions for frozen S2a v2 paths."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Mapping

import yaml

from mirage_kan.artifacts.library import (
    claim_artifact_directory,
    finalize_claimed_directory,
    terminalize_claimed_directory,
)
from mirage_kan.data.pit import sha256_file
from mirage_kan.governance.authority import AuthorityGuard
from mirage_kan.protocol import BASE_LOCK

_DEFAULT_LOCK = BASE_LOCK
_MINING_PARENTS = {
    "mining_run": "artifacts",
    "kan_library": "factor_libraries",
    "gp_control_library": "factor_libraries",
    "permutation_control_library": "factor_libraries",
    "blackbox_control": "controls",
    "mechanism_cards": "mechanism_cards",
    "blind_review_package": "reviews",
}
_DEVELOPMENT_ARMS = (
    "alpha158_replay",
    "kan_e3_selected",
    "typed_gp_sr_control",
    "matched_blackbox_control",
    "kan_e3_permutation_control",
)


class TopologyTransaction:
    """Claim, publish, or terminalize one frozen multi-target topology."""

    def __init__(
        self,
        *,
        workspace: Path,
        protocol_id: str,
        phase: str,
        config_sha256: str,
        targets: Mapping[str, Path],
        top_key: str,
        preclaim_path: Path,
        recovery_path: Path,
    ) -> None:
        self.workspace = workspace
        self.protocol_id = protocol_id
        self.phase = phase
        self.config_sha256 = config_sha256
        self.targets = dict(targets)
        self.top_key = top_key
        self.preclaim_path = preclaim_path
        self.recovery_path = recovery_path
        self.child_keys = tuple(key for key in self.targets if key != top_key)
        topology_record = {
            "protocol_id": protocol_id,
            "phase": phase,
            "config_sha256": config_sha256,
            "top_key": top_key,
            "targets": {
                key: str(path.relative_to(workspace))
                for key, path in self.targets.items()
            },
        }
        self.topology_sha256 = _json_sha256(topology_record)
        self._topology_record = topology_record
        self._owned: set[str] = set()
        self._published: set[str] = set()

    @classmethod
    def from_frozen_config(
        cls,
        workspace: Path | str,
        *,
        phase: str,
        base_lock_path: Path | str = _DEFAULT_LOCK,
    ) -> TopologyTransaction:
        """Load only directory artifacts from the hash-locked v2 config."""
        root = Path(workspace).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("topology workspace is not a directory")
        lock_path = _regular_contained_file(root, base_lock_path, label="base lock")
        lock = _read_mapping_json(lock_path, label="base lock")
        protocol = _mapping(lock, "protocol")
        protocol_id = _string(protocol, "protocol_id")
        config_path = _regular_contained_file(
            root, _string(protocol, "path"), label="frozen config"
        )
        config_sha256 = sha256_file(config_path)
        if config_sha256 != _string(protocol, "sha256"):
            raise ValueError("frozen config hash mismatch")
        config = _read_mapping_yaml(config_path)
        if config.get("protocol_id") != protocol_id:
            raise ValueError("frozen config protocol ID mismatch")
        artifact_paths = _mapping(config, "artifact_paths")
        if phase == "mining":
            raw_targets = {key: _string(artifact_paths, key) for key in _MINING_PARENTS}
            parent_classes = _MINING_PARENTS
            top_key = "mining_run"
            preclaim_key = "mining_preclaim"
            recovery_key = "mining_recovery_receipt"
        elif phase == "development":
            evaluations = _mapping(artifact_paths, "evaluations")
            if set(evaluations) != set(_DEVELOPMENT_ARMS):
                raise ValueError(
                    "development evaluation topology differs from frozen arms"
                )
            raw_targets = {
                "decision_artifact": _string(artifact_paths, "decision_artifact")
            }
            raw_targets.update(
                {
                    f"evaluation:{arm}": _string(evaluations, arm)
                    for arm in _DEVELOPMENT_ARMS
                }
            )
            parent_classes = {key: "evaluations" for key in raw_targets}
            top_key = "decision_artifact"
            preclaim_key = "development_preclaim"
            recovery_key = "development_recovery_receipt"
        else:
            raise ValueError("topology phase must be 'mining' or 'development'")
        targets = {
            key: _artifact_directory_path(
                root, raw_targets[key], expected_parent=parent_classes[key], label=key
            )
            for key in raw_targets
        }
        if len(set(targets.values())) != len(targets):
            raise ValueError("frozen artifact topology contains an alias")
        preclaim_path = _control_json_path(
            root,
            _string(artifact_paths, preclaim_key),
            expected_parent="openings",
            label=preclaim_key,
        )
        recovery_path = _control_json_path(
            root,
            _string(artifact_paths, recovery_key),
            expected_parent="recoveries",
            label=recovery_key,
        )
        return cls(
            workspace=root,
            protocol_id=protocol_id,
            phase=phase,
            config_sha256=config_sha256,
            targets=targets,
            top_key=top_key,
            preclaim_path=preclaim_path,
            recovery_path=recovery_path,
        )

    def preclaim(self) -> dict[str, object]:
        """Consume the topology receipt path before claiming any artifact directory."""
        self._prepare_parents()
        for key, path in self.targets.items():
            if path.exists() or path.is_symlink():
                raise FileExistsError(
                    f"refusing to replace artifact topology target {key}: {path}"
                )
        record = {
            "schema_version": "mirage_topology_preclaim_v2",
            **self._topology_record,
            "topology_sha256": self.topology_sha256,
            "publication_state": "preclaimed",
        }
        _write_json_exclusive(self.preclaim_path, record)
        return record

    def claim_all(self) -> tuple[Path, ...]:
        """Claim every directory; terminalize prior owned claims on any failure."""
        self._verify_preclaim()
        try:
            for key, path in self.targets.items():
                claim_artifact_directory(path)
                self._owned.add(key)
                self._bind_claim_marker(key)
        except BaseException as error:
            try:
                self._terminalize_owned(
                    {"failure_class": "topology_claim_failure", "error": str(error)}
                )
            except BaseException as cleanup_error:
                error.add_note(f"topology cleanup also failed: {cleanup_error}")
            raise
        return tuple(self.targets.values())

    def publish_child(
        self,
        key: str,
        staging_path: Path | str,
        *,
        authority_guard: AuthorityGuard,
        authority_capability: str,
        required_manifest: str = "manifest.json",
    ) -> None:
        """Publish one child, invalidating the complete topology on failure."""
        if key not in self.child_keys:
            raise ValueError(f"not a child topology key: {key}")
        authority_guard.verify_capability(
            authority_capability,
            boundary="before_each_artifact_publication",
            arm=key,
        )
        self._publish(key, staging_path, required_manifest=required_manifest)

    def publish_top_bundle(
        self,
        staging_path: Path | str,
        *,
        authority_guard: AuthorityGuard,
        authority_capability: str,
        final_decision_capability: str | None = None,
        required_manifest: str = "manifest.json",
    ) -> None:
        """Publish the top bundle only after every child manifest is durable."""
        authority_guard.verify_capability(
            authority_capability,
            boundary="before_each_artifact_publication",
            arm=self.top_key,
        )
        if self.phase == "development":
            if final_decision_capability is None:
                raise PermissionError(
                    "development top bundle requires final-decision authority"
                )
            authority_guard.verify_capability(
                final_decision_capability,
                boundary="before_final_decision_publication",
            )
        unpublished = [
            key for key in self.child_keys if not self._is_published(self.targets[key])
        ]
        if unpublished:
            raise RuntimeError("top bundle cannot publish before all children")
        self._publish(self.top_key, staging_path, required_manifest=required_manifest)

    def terminalize(self, payload: Mapping[str, object]) -> dict[str, object]:
        """Invalidate every claimed or published target owned by this transaction."""
        results = self._terminalize_owned(payload)
        return {
            "state": "terminal_failure",
            "topology_sha256": self.topology_sha256,
            "targets": results,
        }

    def recover(self, payload: Mapping[str, object]) -> dict[str, object]:
        """Finish an interrupted topology as terminal; never resume scientific work."""
        self._verify_preclaim()
        if self.recovery_path.is_file() and not self.recovery_path.is_symlink():
            existing = _read_mapping_json(self.recovery_path, label="recovery receipt")
            if existing.get("topology_sha256") != self.topology_sha256:
                raise ValueError("recovery receipt belongs to a different topology")
            self._verify_all_terminal()
            return dict(existing)
        if self.recovery_path.exists() or self.recovery_path.is_symlink():
            raise FileExistsError("invalid recovery receipt path")
        results: dict[str, object] = {}
        for key, path in self.targets.items():
            if not path.exists() and not path.is_symlink():
                claim_artifact_directory(path)
                self._bind_claim_marker(key)
            else:
                self._verify_ownership_evidence(key)
            results[key] = _terminalize_idempotent(
                path, self._terminal_payload(key, payload)
            )
        receipt = {
            "schema_version": "mirage_topology_recovery_v2",
            "protocol_id": self.protocol_id,
            "phase": self.phase,
            "topology_sha256": self.topology_sha256,
            "state": "terminal_failure",
            "targets": results,
        }
        _write_json_exclusive(self.recovery_path, receipt)
        return receipt

    def _publish(
        self, key: str, staging_path: Path | str, *, required_manifest: str
    ) -> None:
        if key not in self._owned:
            path = self.targets[key]
            if not (path / ".INCOMPLETE").is_file():
                raise RuntimeError(f"topology target is not claimed: {key}")
            self._owned.add(key)
        try:
            self._bind_staging_manifest(key, staging_path, required_manifest)
            finalize_claimed_directory(
                staging_path,
                self.targets[key],
                required_manifest=required_manifest,
            )
            self._published.add(key)
        except BaseException as error:
            try:
                self._terminalize_owned(
                    {
                        "failure_class": "topology_publication_failure",
                        "error": str(error),
                    }
                )
            except BaseException as cleanup_error:
                error.add_note(f"topology cleanup also failed: {cleanup_error}")
            raise

    def _terminalize_owned(self, payload: Mapping[str, object]) -> dict[str, object]:
        results: dict[str, object] = {}
        errors: list[BaseException] = []
        for key in self.targets:
            if key not in self._owned:
                continue
            try:
                results[key] = _terminalize_idempotent(
                    self.targets[key], self._terminal_payload(key, payload)
                )
            except BaseException as error:
                errors.append(error)
        if errors:
            raise RuntimeError(
                f"could not terminalize all topology targets ({len(errors)} failures)"
            ) from errors[0]
        return results

    def _prepare_parents(self) -> None:
        parents = {path.parent for path in self.targets.values()}
        parents.update({self.preclaim_path.parent, self.recovery_path.parent})
        for parent in sorted(parents, key=lambda path: len(path.parts)):
            if not parent.exists() and not parent.is_symlink():
                _require_real_directory(
                    self.workspace, parent.parent, label=f"{parent.name} ancestor"
                )
                os.mkdir(parent, 0o700)
                _fsync_directory(parent.parent)
            _require_real_directory(
                self.workspace, parent, label=f"{parent.name} artifact parent"
            )

    def _bind_claim_marker(self, key: str) -> None:
        marker = self.targets[key] / ".INCOMPLETE"
        if not marker.is_file() or marker.is_symlink():
            raise ValueError(f"claimed topology marker is invalid: {key}")
        _rewrite_regular_file(
            marker,
            _canonical_json_bytes(
                {
                    "schema_version": "mirage_topology_claim_v2",
                    "protocol_id": self.protocol_id,
                    "phase": self.phase,
                    "topology_sha256": self.topology_sha256,
                    "topology_key": key,
                }
            ),
        )

    def _bind_staging_manifest(
        self, key: str, staging_path: Path | str, required_manifest: str
    ) -> None:
        staging = Path(staging_path).resolve(strict=True)
        manifest = staging / required_manifest
        if not manifest.is_file() or manifest.is_symlink():
            raise ValueError(f"staging manifest is invalid for topology target: {key}")
        record = _read_mapping_json(manifest, label=f"staging manifest {key}")
        for field, expected in (
            ("topology_sha256", self.topology_sha256),
            ("topology_key", key),
        ):
            observed = record.get(field)
            if observed is not None and observed != expected:
                raise ValueError(f"staging manifest has the wrong {field}: {key}")
        bound = dict(record)
        bound["topology_sha256"] = self.topology_sha256
        bound["topology_key"] = key
        _rewrite_regular_file(manifest, _canonical_json_bytes(bound))

    def _verify_ownership_evidence(self, key: str) -> None:
        path = self.targets[key]
        evidence_paths = (
            path / ".INCOMPLETE",
            path / "manifest.json",
            path / "terminal_failure.json",
        )
        for evidence in evidence_paths:
            if not evidence.is_file() or evidence.is_symlink():
                continue
            try:
                record = _read_mapping_json(evidence, label=f"ownership evidence {key}")
            except ValueError:
                continue
            if (
                record.get("topology_sha256") == self.topology_sha256
                and record.get("topology_key") == key
            ):
                return
        raise ValueError(f"topology target lacks transaction ownership evidence: {key}")

    def _terminal_payload(
        self, key: str, payload: Mapping[str, object]
    ) -> dict[str, object]:
        return {
            **payload,
            "protocol_id": self.protocol_id,
            "phase": self.phase,
            "topology_sha256": self.topology_sha256,
            "topology_key": key,
        }

    def _verify_preclaim(self) -> Mapping[str, object]:
        if not self.preclaim_path.is_file() or self.preclaim_path.is_symlink():
            raise RuntimeError("topology preclaim does not exist")
        record = _read_mapping_json(self.preclaim_path, label="topology preclaim")
        if record.get("topology_sha256") != self.topology_sha256:
            raise ValueError("topology preclaim does not match frozen topology")
        return record

    def _verify_all_terminal(self) -> None:
        for key, path in self.targets.items():
            terminal = path / "terminal_failure.json"
            if not terminal.is_file() or terminal.is_symlink():
                raise ValueError(f"recovery target is not terminal: {key}")
            record = _read_mapping_json(terminal, label=f"terminal target {key}")
            if record.get("publication_state") != "terminal_failure":
                raise ValueError(f"recovery target has invalid terminal state: {key}")
            if (
                record.get("topology_sha256") != self.topology_sha256
                or record.get("topology_key") != key
            ):
                raise ValueError(f"recovery target has invalid ownership: {key}")

    @staticmethod
    def _is_published(path: Path) -> bool:
        return (
            path.is_dir()
            and not path.is_symlink()
            and not (path / ".INCOMPLETE").exists()
            and not (path / "terminal_failure.json").exists()
            and (path / "manifest.json").is_file()
            and not (path / "manifest.json").is_symlink()
        )


def _terminalize_idempotent(
    path: Path, payload: Mapping[str, object]
) -> dict[str, object]:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"terminal topology target is not a real directory: {path}")
    terminal = path / "terminal_failure.json"
    marker = path / ".INCOMPLETE"
    if terminal.is_file() and not terminal.is_symlink():
        record = _read_mapping_json(terminal, label="terminal failure")
        if record.get("publication_state") != "terminal_failure":
            raise ValueError("existing terminal record is not terminal")
        if marker.is_file() and not marker.is_symlink():
            marker.unlink()
            _fsync_directory(path)
            _fsync_directory(path.parent)
        return {
            "path": str(path),
            "state": "terminal_failure",
            "terminal_sha256": sha256_file(terminal),
        }
    return terminalize_claimed_directory(path, payload, invalidate_published=True)


def _artifact_directory_path(
    root: Path,
    raw_path: str,
    *,
    expected_parent: str,
    label: str,
) -> Path:
    raw = Path(raw_path)
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"{label} artifact path escapes the workspace")
    if len(raw.parts) != 2 or raw.parts[0] != expected_parent:
        raise ValueError(
            f"{label} artifact must be a direct child of {expected_parent}"
        )
    parent = root / expected_parent
    _allow_missing_real_directory(root, parent, label=f"{label} parent")
    target = parent / raw.name
    if target.is_symlink():
        raise ValueError(f"{label} artifact path is a symlink")
    return target


def _control_json_path(
    root: Path,
    raw_path: str,
    *,
    expected_parent: str,
    label: str,
) -> Path:
    raw = Path(raw_path)
    expected_parts = ("governance", expected_parent)
    if (
        raw.is_absolute()
        or ".." in raw.parts
        or len(raw.parts) != 3
        or raw.parts[:2] != expected_parts
        or raw.suffix != ".json"
    ):
        raise ValueError(f"{label} must be a direct governance JSON control file")
    parent = root / "governance" / expected_parent
    _allow_missing_real_directory(root, parent, label=f"{label} parent")
    path = parent / raw.name
    if path.is_symlink():
        raise ValueError(f"{label} control path is a symlink")
    return path


def _regular_contained_file(root: Path, raw_path: Path | str, *, label: str) -> Path:
    raw = Path(raw_path)
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"{label} path escapes the workspace")
    path = root.joinpath(raw)
    resolved_parent = path.parent.resolve(strict=True)
    if not resolved_parent.is_relative_to(root):
        raise ValueError(f"{label} path escapes the workspace")
    _require_real_directory(root, path.parent, label=f"{label} parent")
    if path.is_symlink():
        raise ValueError(f"{label} path is a symlink")
    state = path.stat()
    if not stat.S_ISREG(state.st_mode):
        raise ValueError(f"{label} path is not a regular file")
    return path


def _require_real_directory(root: Path, path: Path, *, label: str) -> None:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError(f"{label} escapes the workspace")
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink")
    if not path.is_dir():
        raise ValueError(f"{label} is not a directory")


def _allow_missing_real_directory(root: Path, path: Path, *, label: str) -> None:
    if path.exists() or path.is_symlink():
        _require_real_directory(root, path, label=label)
        return
    _require_real_directory(root, path.parent, label=f"{label} ancestor")


def _mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    child = value.get(key)
    if not isinstance(child, Mapping):
        raise ValueError(f"frozen topology lacks mapping: {key}")
    return child


def _string(value: Mapping[str, object], key: str) -> str:
    child = value.get(key)
    if not isinstance(child, str) or not child:
        raise ValueError(f"frozen topology lacks string: {key}")
    return child


def _read_mapping_json(path: Path, *, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _read_mapping_yaml(path: Path) -> Mapping[str, object]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError("frozen config is unreadable") from error
    if not isinstance(value, Mapping):
        raise ValueError("frozen config is not a YAML mapping")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _write_json_exclusive(path: Path, value: object) -> None:
    data = _canonical_json_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("exclusive topology write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _rewrite_regular_file(path: Path, data: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"topology binding target is not a regular file: {path}")
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("topology binding write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
