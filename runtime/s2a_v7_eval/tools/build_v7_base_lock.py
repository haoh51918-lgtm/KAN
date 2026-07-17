"""Build the prospective v7 base lock without loading scientific labels."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


SOURCE_LOCK = Path("prereg/s2a_kan_e3_vertical_v6.lock.json")
SOURCE_CONFIG = Path("configs/experiments/s2a_kan_e3_vertical_v6.yaml")
SOURCE_IMPLEMENTATION = Path(
    "prereg/s2a_kan_e3_vertical_v6_implementation.lock.json"
)
TARGET_CONFIG = Path("configs/experiments/s2a_kan_e3_vertical_v7.yaml")
TARGET_PREREGISTRATION = Path("prereg/s2a_kan_e3_vertical_v7.md")
TARGET_INCIDENT = Path(
    "governance/incidents/2026-07-17_s2a_v7_preopening_rebind_successor.md"
)
TARGET_DIRECTIVE = Path(
    "governance/decisions/2026-07-17_review_adoption_and_living_manual.md"
)
OUTPUT = Path("prereg/s2a_kan_e3_vertical_v7.lock.json")
SOURCE_PROTOCOL = "s2a_kan_e3_vertical_v6"
TARGET_PROTOCOL = "s2a_kan_e3_vertical_v7"

SOURCE_IDENTITY_FILES = (
    SOURCE_LOCK,
    SOURCE_CONFIG,
    SOURCE_IMPLEMENTATION,
    Path("prereg/s2a_kan_e3_vertical_v6.md"),
    Path("governance/incidents/2026-07-17_s2a_v6_lineage_corrective_successor.md"),
    Path("governance/openings/s2a_kan_e3_vertical_v6_mining_preclaim.json"),
    Path("governance/openings/s2a_kan_e3_vertical_v6_mining.json"),
    Path("reports/s2a_v6_implementation_lock_checkpoint.md"),
    Path("reports/s2a_v6_mining_and_preopening_rehearsal.md"),
    Path("Review-from-claude.md"),
    Path("docs/research/MIRAGE_KAN_LIVING_MANUAL.md"),
    Path("governance/feedback/2026-07-17_review_from_claude_1_triage.md"),
    Path("governance/decisions/2026-07-17_graph_core_unlock.md"),
    TARGET_DIRECTIVE,
)
SOURCE_DIRECTORIES = (
    Path("artifacts/s2a_kan_e3_mining_v6"),
    Path("factor_libraries/s2a_kan_e3_selected_v6"),
    Path("factor_libraries/s2a_typed_gp_sr_control_v6"),
    Path("factor_libraries/s2a_kan_e3_permutation_control_v6"),
    Path("controls/s2a_matched_blackbox_v6"),
    Path("mechanism_cards/s2a_kan_e3_selected_v6"),
    Path("reviews/s2a_kan_e3_blind_v6"),
    Path("governance/authority/s2a_kan_e3_vertical_v6"),
)
RUNTIME_FILES = (
    Path("runtime/s2a_v7_eval/tools/build_v7_config.py"),
    Path("runtime/s2a_v7_eval/tools/build_v7_base_lock.py"),
    Path("evaluations/runtime/s2a_v7_tracking/README.md"),
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
        raise ValueError(f"required regular file is missing: {relative}")
    if not path.resolve(strict=True).is_relative_to(root):
        raise ValueError(f"file escapes workspace: {relative}")
    return path


def _directory_files(root: Path, relative: Path) -> list[Path]:
    directory = root / relative
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"required real directory is missing: {relative}")
    files: list[Path] = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"custody directory contains a symlink: {path}")
        if path.is_file():
            files.append(path.relative_to(root))
    if not files:
        raise ValueError(f"custody directory is empty: {relative}")
    return files


def _read_json(root: Path, relative: Path) -> dict[str, Any]:
    value = json.loads(_regular_file(root, relative).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {relative}")
    return value


def _read_yaml(root: Path, relative: Path) -> dict[str, Any]:
    value = yaml.safe_load(_regular_file(root, relative).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML mapping required: {relative}")
    return value


def _scientific_payload(value: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(value)
    payload.pop("protocol_id", None)
    payload.pop("artifact_paths", None)
    payload.pop("mining_source", None)
    payload["claim_boundary"].pop("graph_unlock_allowed", None)
    return payload


def _custody_files(root: Path) -> dict[str, str]:
    paths = set(SOURCE_IDENTITY_FILES)
    for directory in SOURCE_DIRECTORIES:
        paths.update(_directory_files(root, directory))
    return {
        relative.as_posix(): _sha256(_regular_file(root, relative))
        for relative in sorted(paths)
    }


def build_base_lock(workspace: Path | str) -> dict[str, Any]:
    root = Path(workspace).resolve(strict=True)
    source_config = _read_yaml(root, SOURCE_CONFIG)
    target_config = _read_yaml(root, TARGET_CONFIG)
    if (
        source_config.get("protocol_id") != SOURCE_PROTOCOL
        or target_config.get("protocol_id") != TARGET_PROTOCOL
        or _scientific_payload(source_config) != _scientific_payload(target_config)
    ):
        raise ValueError("v7 mining-affecting scientific configuration differs")
    if target_config["claim_boundary"].get("graph_unlock_allowed") is not True:
        raise ValueError("v7 does not contain the authorized graph unlock")
    lock = copy.deepcopy(_read_json(root, SOURCE_LOCK))
    lock["evidence_class"] = "corrective_adaptive_repeated_development_screen"
    lock["formal_promotion_allowed"] = False
    lock["test_period_role"] = "corrective_adaptive_repeated_development_only"
    governance = lock["governance"]
    governance["active_directive_path"] = TARGET_DIRECTIVE.as_posix()
    governance["active_directive_sha256"] = _sha256(
        _regular_file(root, TARGET_DIRECTIVE)
    )
    governance["supersession_incident_path"] = TARGET_INCIDENT.as_posix()
    governance["supersession_incident_sha256"] = _sha256(
        _regular_file(root, TARGET_INCIDENT)
    )
    governance["v6_candidate_membership_reuse_allowed"] = True
    governance["v6_reuse_scope"] = "exact_immutable_whole_topology"
    governance["v6_disposition"] = "successful_mining_development_unopened"
    lock["predecessor_custody"] = {
        "protocol_id": SOURCE_PROTOCOL,
        "scientific_observation": (
            "successful_mining_development_unopened_exact_rebind"
        ),
        "candidate_membership_reuse_allowed": True,
        "reuse_scope": "exact_immutable_whole_topology",
        "new_label_entitlement_allowed": False,
        "reselection_allowed": False,
        "reordering_allowed": False,
        "retuning_allowed": False,
        "files": _custody_files(root),
    }
    lock["preregistration"] = {
        "path": TARGET_PREREGISTRATION.as_posix(),
        "sha256": _sha256(_regular_file(root, TARGET_PREREGISTRATION)),
    }
    lock["protocol"] = {
        "path": TARGET_CONFIG.as_posix(),
        "protocol_id": TARGET_PROTOCOL,
        "sha256": _sha256(_regular_file(root, TARGET_CONFIG)),
    }
    lock["runtime"] = {
        "files": {
            relative.as_posix(): _sha256(_regular_file(root, relative))
            for relative in RUNTIME_FILES
        }
    }
    if set(lock["runtime"]["files"]) & set(lock["predecessor_custody"]["files"]):
        raise ValueError("v7 runtime aliases predecessor custody")
    return lock


def _body(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("digest", "write", "verify"))
    parser.add_argument("--workspace", default=".")
    args = parser.parse_args()
    root = Path(args.workspace).resolve(strict=True)
    payload = build_base_lock(root)
    body = _body(payload)
    output = root / OUTPUT
    if args.mode == "write":
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            output.unlink(missing_ok=True)
            raise
    elif args.mode == "verify" and output.read_bytes() != body:
        raise ValueError("v7 base lock differs from its deterministic source build")
    print(
        json.dumps(
            {
                "custody_files": len(payload["predecessor_custody"]["files"]),
                "runtime_files": len(payload["runtime"]["files"]),
                "sha256": hashlib.sha256(body).hexdigest(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
