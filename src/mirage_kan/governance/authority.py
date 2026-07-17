"""Live protocol-identity revalidation at frozen scientific boundaries."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from mirage_kan.data.pit import sha256_file
from mirage_kan.protocol import BASE_LOCK

FROZEN_BOUNDARIES = (
    "before_first_label_access",
    "before_each_scientific_or_control_arm",
    "before_each_artifact_publication",
    "before_development_opening",
    "before_final_decision_publication",
)
_LABELED_BOUNDARIES = {
    "before_each_scientific_or_control_arm",
    "before_each_artifact_publication",
}
_ALLOWED_PREDECESSOR_OBSERVATIONS = frozenset(
    {
        "none",
        "pre_development_admission_count_only",
        "inconclusive_infrastructure_with_quarantined_development_outputs",
        "successful_mining_development_unopened_exact_rebind",
        "terminal_preopening_rebind_software_failure",
    }
)
_DEFAULT_LOCK = BASE_LOCK


class AuthoritySuperseded(RuntimeError):
    """The live protocol identity no longer matches its prospective lock."""


@dataclass(frozen=True)
class AuthorityReceipt:
    """An immutable receipt issued by one live ``AuthorityGuard`` instance."""

    schema_version: str
    protocol_id: str
    guard_instance_sha256: str
    base_lock_sha256: str
    sequence: int
    boundary: str
    arm: str | None
    authority_files: tuple[tuple[str, str], ...]
    authority_sha256: str
    checked_at_unix_ns: int
    receipt_sha256: str
    capability: str

    def as_dict(self) -> dict[str, object]:
        """Return a canonical-JSON-compatible representation."""
        return {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "guard_instance_sha256": self.guard_instance_sha256,
            "base_lock_sha256": self.base_lock_sha256,
            "sequence": self.sequence,
            "boundary": self.boundary,
            "arm": self.arm,
            "authority_files": dict(self.authority_files),
            "authority_sha256": self.authority_sha256,
            "checked_at_unix_ns": self.checked_at_unix_ns,
            "receipt_sha256": self.receipt_sha256,
            "capability": self.capability,
        }


class AuthorityGuard:
    """Revalidate five frozen live-authority boundaries and issue capabilities."""

    def __init__(
        self,
        workspace: Path | str,
        *,
        base_lock_path: Path | str = _DEFAULT_LOCK,
        ledger_path: Path | str | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve(strict=True)
        if not self.workspace.is_dir():
            raise ValueError("authority workspace is not a directory")
        self.base_lock_path = _contained_regular_file(
            self.workspace, base_lock_path, label="base_lock"
        )
        self.base_lock_sha256 = sha256_file(self.base_lock_path)
        self._base_lock = _read_json(self.base_lock_path, label="base_lock")
        self.protocol_id = _required_string(
            _required_mapping(self._base_lock, "protocol"), "protocol_id"
        )
        protocol_path = _required_string(
            _required_mapping(self._base_lock, "protocol"), "path"
        )
        self._authority_specs = self._load_authority_specs(protocol_path)
        config_path = self._authority_specs["config"][0]
        config = _read_yaml(config_path)
        boundaries = tuple(
            _required_mapping(config, "authority_revalidation").get("boundaries", ())
        )
        if boundaries != FROZEN_BOUNDARIES:
            raise AuthoritySuperseded(
                "config authority boundaries differ from the frozen five-boundary contract"
            )
        arms = _required_mapping(config, "controls").get("arms")
        if not isinstance(arms, list) or not all(
            isinstance(arm, str) and arm for arm in arms
        ):
            raise AuthoritySuperseded("config scientific arms are invalid")
        self._scientific_arms = frozenset(arms)
        self._nonce = os.urandom(32)
        self.guard_instance_sha256 = hashlib.sha256(self._nonce).hexdigest()
        default_ledger = (
            Path("governance") / "authority" / self.protocol_id / "revalidation.jsonl"
        )
        self.ledger_path = _contained_output_path(
            self.workspace,
            default_ledger if ledger_path is None else ledger_path,
            label="authority ledger",
        )
        self._sequence_directory = self.ledger_path.parent / "receipts"
        self._issued: dict[str, AuthorityReceipt] = {}
        self._verify_live_authority()

    def revalidate(self, boundary: str, arm: str | None = None) -> AuthorityReceipt:
        """Fail closed on drift, otherwise append one immutable boundary receipt."""
        self._validate_boundary(boundary, arm)
        authority_files, authority_sha256 = self._verify_live_authority()
        sequence = self._allocate_sequence()
        payload = {
            "schema_version": "mirage_authority_receipt_v2",
            "protocol_id": self.protocol_id,
            "guard_instance_sha256": self.guard_instance_sha256,
            "base_lock_sha256": self.base_lock_sha256,
            "sequence": sequence,
            "boundary": boundary,
            "arm": arm,
            "authority_files": dict(authority_files),
            "authority_sha256": authority_sha256,
            "checked_at_unix_ns": time.time_ns(),
        }
        receipt_sha256 = _json_sha256(payload)
        capability = hashlib.sha256(
            self._nonce
            + bytes.fromhex(receipt_sha256)
            + boundary.encode("utf-8")
            + b"\0"
            + (arm or "").encode("utf-8")
        ).hexdigest()
        receipt = AuthorityReceipt(
            schema_version="mirage_authority_receipt_v2",
            protocol_id=self.protocol_id,
            guard_instance_sha256=self.guard_instance_sha256,
            base_lock_sha256=self.base_lock_sha256,
            sequence=sequence,
            boundary=boundary,
            arm=arm,
            authority_files=authority_files,
            authority_sha256=authority_sha256,
            checked_at_unix_ns=int(payload["checked_at_unix_ns"]),
            receipt_sha256=receipt_sha256,
            capability=capability,
        )
        receipt_bytes = _canonical_json_bytes(receipt.as_dict())
        sequence_path = self._sequence_directory / f"{sequence:020d}.json"
        _write_exclusive(sequence_path, receipt_bytes)
        _append_durable(self.ledger_path, _canonical_json_line_bytes(receipt.as_dict()))
        self._issued[capability] = receipt
        return receipt

    def verify_capability(
        self,
        capability: str,
        *,
        boundary: str,
        arm: str | None = None,
    ) -> AuthorityReceipt:
        """Verify an issued capability against its boundary and current authority."""
        self._verify_live_authority()
        if not isinstance(capability, str):
            raise TypeError("authority verifier requires a capability string")
        receipt = self._issued.get(capability)
        if receipt is None:
            raise ValueError("unknown capability for this AuthorityGuard instance")
        if receipt.boundary != boundary or receipt.arm != arm:
            raise ValueError("capability has the wrong boundary or arm")
        expected = hashlib.sha256(
            self._nonce
            + bytes.fromhex(receipt.receipt_sha256)
            + boundary.encode("utf-8")
            + b"\0"
            + (arm or "").encode("utf-8")
        ).hexdigest()
        if not _constant_time_equal(expected, capability):
            raise ValueError("authority capability is invalid")
        return receipt

    def _load_authority_specs(self, protocol_path: str) -> dict[str, tuple[Path, str]]:
        proposal = _required_mapping(self._base_lock, "proposal")
        preregistration = _required_mapping(self._base_lock, "preregistration")
        governance = _required_mapping(self._base_lock, "governance")
        if proposal.get("authority") not in {"sole_proposal_authority", "idea_draft"}:
            raise AuthoritySuperseded("proposal lock has an unknown document role")
        if proposal.get("path") != "KAN_Alpha_PR.md":
            raise AuthoritySuperseded(
                "the live proposal must be workspace-root KAN_Alpha_PR.md; archives cannot substitute"
            )
        raw_specs = {
            "proposal": (proposal.get("path"), proposal.get("sha256")),
            "config": (
                protocol_path,
                _required_mapping(self._base_lock, "protocol").get("sha256"),
            ),
            "preregistration": (
                preregistration.get("path"),
                preregistration.get("sha256"),
            ),
            "directive": (
                governance.get("active_directive_path"),
                governance.get("active_directive_sha256"),
            ),
            "incident": (
                governance.get("supersession_incident_path"),
                governance.get("supersession_incident_sha256"),
            ),
        }
        custody = self._base_lock.get("predecessor_custody")
        if custody is not None:
            if not isinstance(custody, Mapping):
                raise AuthoritySuperseded("predecessor custody is invalid")
            predecessor_protocol = custody.get("protocol_id")
            if (
                not isinstance(predecessor_protocol, str)
                or not predecessor_protocol
                or predecessor_protocol == self.protocol_id
                or custody.get("scientific_observation")
                not in _ALLOWED_PREDECESSOR_OBSERVATIONS
            ):
                raise AuthoritySuperseded("predecessor custody disposition is invalid")
            custody_files = custody.get("files")
            if not isinstance(custody_files, Mapping) or not custody_files:
                raise AuthoritySuperseded("predecessor custody files are invalid")
            for index, raw_path in enumerate(sorted(custody_files)):
                raw_specs[f"predecessor_{index:02d}"] = (
                    raw_path,
                    custody_files[raw_path],
                )
        specs: dict[str, tuple[Path, str]] = {}
        for name, (raw_path, expected_hash) in raw_specs.items():
            if not isinstance(raw_path, str) or not isinstance(expected_hash, str):
                raise AuthoritySuperseded(f"{name} authority identity is missing")
            try:
                path = _contained_regular_file(self.workspace, raw_path, label=name)
            except (FileNotFoundError, ValueError) as error:
                raise AuthoritySuperseded(
                    f"{name} live authority is invalid"
                ) from error
            specs[name] = (path, expected_hash)
        return specs

    def _verify_live_authority(self) -> tuple[tuple[tuple[str, str], ...], str]:
        try:
            _assert_live_regular(self.workspace, self.base_lock_path, "base_lock")
            current_lock_hash = sha256_file(self.base_lock_path)
        except (OSError, ValueError) as error:
            raise AuthoritySuperseded(
                "base_lock live authority is unreadable"
            ) from error
        if current_lock_hash != self.base_lock_sha256:
            raise AuthoritySuperseded("base_lock live hash mismatch")
        observed: list[tuple[str, str]] = []
        for name, (path, expected_hash) in self._authority_specs.items():
            try:
                _assert_live_regular(self.workspace, path, name)
                actual_hash = sha256_file(path)
            except (OSError, ValueError) as error:
                raise AuthoritySuperseded(
                    f"{name} live authority is unreadable"
                ) from error
            if actual_hash != expected_hash:
                raise AuthoritySuperseded(f"{name} live hash mismatch")
            observed.append((name, actual_hash))
        authority_files = tuple(observed)
        authority_sha256 = _json_sha256(dict(authority_files))
        return authority_files, authority_sha256

    def _validate_boundary(self, boundary: str, arm: str | None) -> None:
        if boundary not in FROZEN_BOUNDARIES:
            raise ValueError(f"authority boundary is not frozen: {boundary!r}")
        if boundary in _LABELED_BOUNDARIES:
            if not isinstance(arm, str) or not arm:
                raise ValueError(
                    f"authority boundary {boundary} requires an arm or publisher"
                )
            if (
                boundary == "before_each_scientific_or_control_arm"
                and arm not in self._scientific_arms
            ):
                raise ValueError(f"unknown scientific arm: {arm}")
        elif arm is not None:
            raise ValueError(f"authority boundary {boundary} does not accept an arm")

    def _allocate_sequence(self) -> int:
        self._sequence_directory.mkdir(parents=True, exist_ok=True)
        for sequence in range(1, 10**9):
            claim = self._sequence_directory / f".{sequence:020d}.claim"
            try:
                descriptor = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
            except FileExistsError:
                continue
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            _fsync_directory(self._sequence_directory)
            return sequence
        raise RuntimeError("authority receipt sequence space exhausted")


def _required_mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    child = value.get(key)
    if not isinstance(child, Mapping):
        raise AuthoritySuperseded(f"base lock/config lacks mapping: {key}")
    return child


def _required_string(value: Mapping[str, object], key: str) -> str:
    child = value.get(key)
    if not isinstance(child, str) or not child:
        raise AuthoritySuperseded(f"base lock lacks string: {key}")
    return child


def _contained_path(root: Path, raw_path: Path | str, *, label: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} path escapes the workspace")
    candidate = root.joinpath(path)
    resolved_parent = candidate.parent.resolve(strict=True)
    if not resolved_parent.is_relative_to(root):
        raise ValueError(f"{label} path escapes the workspace")
    _reject_symlink_components(root, candidate.parent)
    return resolved_parent / candidate.name


def _contained_output_path(root: Path, raw_path: Path | str, *, label: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} path escapes the workspace")
    current = root
    for part in path.parts[:-1]:
        current = current / part
        if current.exists() or current.is_symlink():
            if current.is_symlink():
                raise ValueError(f"{label} path contains a symlink")
            if not current.is_dir():
                raise ValueError(f"{label} parent is not a directory")
            if not current.resolve(strict=True).is_relative_to(root):
                raise ValueError(f"{label} path escapes the workspace")
    return root.joinpath(path)


def _contained_regular_file(root: Path, raw_path: Path | str, *, label: str) -> Path:
    candidate = _contained_path(root, raw_path, label=label)
    if candidate.is_symlink():
        raise ValueError(f"{label} path is a symlink")
    state = candidate.stat()
    if not stat.S_ISREG(state.st_mode):
        raise ValueError(f"{label} path is not a regular file")
    return candidate


def _reject_symlink_components(root: Path, parent: Path) -> None:
    relative = parent.relative_to(root)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"path contains a symlinked component: {current}")


def _assert_live_regular(root: Path, path: Path, label: str) -> None:
    _reject_symlink_components(root, path.parent)
    if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"{label} live path is not a regular file")


def _read_json(path: Path, *, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuthoritySuperseded(f"{label} is unreadable") from error
    if not isinstance(value, Mapping):
        raise AuthoritySuperseded(f"{label} is not a JSON object")
    return value


def _read_yaml(path: Path) -> Mapping[str, object]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise AuthoritySuperseded("config authority is unreadable") from error
    if not isinstance(value, Mapping):
        raise AuthoritySuperseded("config authority is not a YAML mapping")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _canonical_json_line_bytes(value: object) -> bytes:
    return (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        _write_all(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _append_durable(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o444)
    try:
        written = os.write(descriptor, data)
        if written != len(data):
            raise OSError("authority ledger append was not atomic")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("exclusive authority write made no progress")
        view = view[written:]


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _constant_time_equal(left: str, right: str) -> bool:
    return hmac.compare_digest(left, right)
