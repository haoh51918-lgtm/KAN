"""Build the immutable v5 base lock without accessing scientific labels."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


V4_PROTOCOL = "s2a_kan_e3_vertical_v4"
V5_PROTOCOL = "s2a_kan_e3_vertical_v5"
OUTPUT = Path("prereg/s2a_kan_e3_vertical_v5.lock.json")

V4_CONFIG = Path("configs/experiments/s2a_kan_e3_vertical_v4.yaml")
V5_CONFIG = Path("configs/experiments/s2a_kan_e3_vertical_v5.yaml")
V4_BASE_LOCK = Path("prereg/s2a_kan_e3_vertical_v4.lock.json")
V4_IMPLEMENTATION_LOCK = Path(
    "prereg/s2a_kan_e3_vertical_v4_implementation.lock.json"
)
V4_PREREGISTRATION = Path("prereg/s2a_kan_e3_vertical_v4.md")
V5_PREREGISTRATION = Path("prereg/s2a_kan_e3_vertical_v5.md")
V5_INCIDENT = Path(
    "governance/incidents/2026-07-17_s2a_v5_calendar_corrective_successor.md"
)

V4_CUSTODY_FILES = (
    V4_CONFIG,
    V4_BASE_LOCK,
    V4_IMPLEMENTATION_LOCK,
    V4_PREREGISTRATION,
    Path("governance/incidents/2026-07-17_s2a_v4_adaptive_successor.md"),
    Path("governance/openings/s2a_kan_e3_vertical_v4_mining_preclaim.json"),
    Path("governance/openings/s2a_kan_e3_vertical_v4_mining.json"),
    Path("governance/openings/s2a_kan_e3_vertical_v4_development_preclaim.json"),
    Path("governance/openings/s2a_kan_e3_vertical_v4_development.json"),
    Path("reports/s2a_v4_implementation_lock_checkpoint.md"),
    Path("reports/s2a_v4_mining_checkpoint.md"),
    Path("reports/s2a_v4_development_calendar_blocker.md"),
)

V4_CUSTODY_DIRECTORIES = (
    Path("artifacts/s2a_kan_e3_mining_v4"),
    Path("controls/s2a_matched_blackbox_v4"),
    Path("factor_libraries/s2a_kan_e3_selected_v4"),
    Path("factor_libraries/s2a_typed_gp_sr_control_v4"),
    Path("factor_libraries/s2a_kan_e3_permutation_control_v4"),
    Path("mechanism_cards/s2a_kan_e3_selected_v4"),
    Path("reviews/s2a_kan_e3_blind_v4"),
    Path("governance/authority/s2a_kan_e3_vertical_v4"),
    Path("evaluations/runtime/s2a_v4_tracking"),
    Path("evaluations/s2a_v4_alpha158_replay"),
    Path("evaluations/s2a_v4_kan_e3_selected"),
    Path("evaluations/s2a_v4_typed_gp_sr_control"),
    Path("evaluations/s2a_v4_matched_blackbox_control"),
    Path("evaluations/s2a_v4_kan_e3_permutation_control"),
    Path("evaluations/s2a_kan_e3_vertical_v4_decision"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(root: Path, relative: Path) -> Path:
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required regular file is missing: {relative.as_posix()}")
    if not path.resolve(strict=True).is_relative_to(root):
        raise ValueError(f"file escapes workspace: {relative.as_posix()}")
    return path


def _regular_directory(root: Path, relative: Path) -> Path:
    path = root / relative
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"required real directory is missing: {relative.as_posix()}")
    if not path.resolve(strict=True).is_relative_to(root):
        raise ValueError(f"directory escapes workspace: {relative.as_posix()}")
    return path


def _directory_files(root: Path, relative: Path) -> list[Path]:
    directory = _regular_directory(root, relative)
    files: list[Path] = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"custody/runtime symlink is forbidden: {path}")
        if path.is_file():
            files.append(path.relative_to(root))
    if not files:
        raise ValueError(f"required directory is empty: {relative.as_posix()}")
    return files


def _read_json(root: Path, relative: Path) -> dict[str, Any]:
    value = json.loads(_regular_file(root, relative).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {relative.as_posix()}")
    return value


def _read_yaml(root: Path, relative: Path) -> dict[str, Any]:
    value = yaml.safe_load(_regular_file(root, relative).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML mapping required: {relative.as_posix()}")
    return value


def _assert_scientific_config_unchanged(root: Path) -> None:
    v4 = _read_yaml(root, V4_CONFIG)
    v5 = copy.deepcopy(_read_yaml(root, V5_CONFIG))
    if v5.get("protocol_id") != V5_PROTOCOL:
        raise ValueError("v5 config has the wrong protocol ID")
    if v5.get("evidence_class") != "corrective_adaptive_repeated_development_screen":
        raise ValueError("v5 config has the wrong evidence class")
    v5["protocol_id"] = V4_PROTOCOL
    v5["evidence_class"] = "prospective_development_screen"
    artifact_paths = v5.get("artifact_paths")
    if not isinstance(artifact_paths, dict):
        raise ValueError("v5 artifact_paths must be a mapping")

    def rewrite(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, str):
            return value.replace("_v5", "_v4")
        return value

    v5["artifact_paths"] = rewrite(artifact_paths)
    if v5 != v4:
        raise ValueError(
            "v5 changes scientific fields beyond protocol, writable paths, and evidence class"
        )


def _custody_files(root: Path, v4_lock: dict[str, Any]) -> dict[str, str]:
    predecessor = v4_lock.get("predecessor_custody")
    if not isinstance(predecessor, dict) or not isinstance(
        predecessor.get("files"), dict
    ):
        raise ValueError("v4 predecessor custody is missing")
    relative_files = {Path(value) for value in predecessor["files"]}
    relative_files.update(V4_CUSTODY_FILES)
    for directory in V4_CUSTODY_DIRECTORIES:
        relative_files.update(_directory_files(root, directory))
    return {
        relative.as_posix(): _sha256(_regular_file(root, relative))
        for relative in sorted(relative_files)
    }


def _runtime_files(root: Path, v4_lock: dict[str, Any]) -> dict[str, str]:
    runtime = v4_lock.get("runtime")
    if not isinstance(runtime, dict) or not isinstance(runtime.get("files"), dict):
        raise ValueError("v4 runtime inventory is missing")
    relative_files = {
        Path(value)
        for value in runtime["files"]
        if value != "evaluations/runtime/s2a_v4_tracking/README.md"
    }
    relative_files.update(_directory_files(root, Path("runtime/s2a_v5_eval")))
    relative_files.add(Path("evaluations/runtime/s2a_v5_tracking/README.md"))
    return {
        relative.as_posix(): _sha256(_regular_file(root, relative))
        for relative in sorted(relative_files)
    }


def build_base_lock(workspace: Path | str) -> dict[str, Any]:
    root = Path(workspace).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("workspace must be a directory")
    _assert_scientific_config_unchanged(root)
    v4_lock = _read_json(root, V4_BASE_LOCK)
    if v4_lock.get("schema_version") != "mirage_s2_prereg_lock_v2":
        raise ValueError("v4 base lock schema is unsupported")
    lock = copy.deepcopy(v4_lock)
    lock["evidence_class"] = "corrective_adaptive_repeated_development_screen"
    lock["formal_promotion_allowed"] = False
    lock["test_period_role"] = "corrective_adaptive_repeated_development_only"
    governance = lock["governance"]
    governance["supersession_incident_path"] = V5_INCIDENT.as_posix()
    governance["supersession_incident_sha256"] = _sha256(
        _regular_file(root, V5_INCIDENT)
    )
    governance["v4_candidate_membership_reuse_allowed"] = False
    governance["v4_disposition"] = (
        "terminal_development_infrastructure_failure_with_quarantined_outputs"
    )
    lock["predecessor_custody"] = {
        "candidate_membership_reuse_allowed": False,
        "files": _custody_files(root, v4_lock),
        "protocol_id": V4_PROTOCOL,
        "scientific_observation": (
            "inconclusive_infrastructure_with_quarantined_development_outputs"
        ),
    }
    lock["preregistration"] = {
        "path": V5_PREREGISTRATION.as_posix(),
        "sha256": _sha256(_regular_file(root, V5_PREREGISTRATION)),
    }
    lock["protocol"] = {
        "path": V5_CONFIG.as_posix(),
        "protocol_id": V5_PROTOCOL,
        "sha256": _sha256(_regular_file(root, V5_CONFIG)),
    }
    lock["runtime"] = {"files": _runtime_files(root, v4_lock)}
    overlap = set(lock["predecessor_custody"]["files"]) & set(
        lock["runtime"]["files"]
    )
    if overlap:
        raise ValueError(f"custody/runtime path overlap: {sorted(overlap)}")
    return lock


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_exclusive(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("digest", "write", "verify"))
    parser.add_argument("--workspace", default=".")
    args = parser.parse_args()
    root = Path(args.workspace).resolve(strict=True)
    payload = build_base_lock(root)
    content = _canonical_bytes(payload)
    expected = hashlib.sha256(content).hexdigest()
    output = root / OUTPUT
    if args.mode == "write":
        _write_exclusive(output, content)
    elif args.mode == "verify":
        if output.is_symlink() or not output.is_file():
            raise ValueError("v5 base lock is not a regular file")
        if output.read_bytes() != content:
            raise ValueError("v5 base lock differs from the live deterministic build")
    print(
        json.dumps(
            {
                "custody_files": len(payload["predecessor_custody"]["files"]),
                "mode": args.mode,
                "path": OUTPUT.as_posix(),
                "runtime_files": len(payload["runtime"]["files"]),
                "sha256": expected,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
